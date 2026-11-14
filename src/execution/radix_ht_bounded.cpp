#include "duckdb/execution/radix_ht_bounded.hpp"

#include "duckdb/common/radix_partitioning.hpp"
#include "duckdb/common/row_operations/row_operations.hpp"
#include "duckdb/common/types/row/partitioned_tuple_data.hpp"
#include "duckdb/common/types/row/tuple_data_iterator.hpp"
#include "duckdb/execution/aggregate_hashtable.hpp"
#include "duckdb/execution/ht_entry.hpp"
#include "duckdb/execution/radix_ht_sink_state.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/storage/buffer_manager.hpp"

#include <cmath>

namespace duckdb {

//! Partitions rows of a node by the radix bits that follow the node's prefix
class BoundedChildPartitionedTupleData : public PartitionedTupleData {
public:
	BoundedChildPartitionedTupleData(BufferManager &buffer_manager, shared_ptr<TupleDataLayout> layout_ptr_p,
	                                 idx_t node_radix_bits, idx_t child_bits, QueryContext context)
	    : PartitionedTupleData(PartitionedTupleDataType::RADIX, buffer_manager, layout_ptr_p, MemoryTag::HASH_TABLE,
	                           context),
	      shift(RadixPartitioning::Shift(node_radix_bits + child_bits)),
	      mask(RadixPartitioning::NumberOfPartitions(child_bits) - 1), hash_col_idx(layout.ColumnCount() - 1) {
		D_ASSERT(node_radix_bits + child_bits <= RadixPartitioning::MAX_RADIX_BITS);
		// The children are finalized by different threads
		separate_allocators = true;
		for (idx_t i = 0; i <= mask; i++) {
			partitions.emplace_back(CreatePartitionCollection());
			partitions.back()->SetPartitionIndex(i);
		}
	}

protected:
	void ComputePartitionIndices(Vector &row_locations, idx_t count, Vector &partition_indices,
	                             unique_ptr<Vector> &utility_vector, const SelectionVector &append_sel) const override {
		if (!utility_vector) {
			utility_vector = make_uniq<Vector>(LogicalType::HASH);
		}
		Vector &hashes = *utility_vector;
		hashes.SetVectorType(VectorType::FLAT_VECTOR);
		partitions[0]->Gather(row_locations, append_sel, count, hash_col_idx, hashes,
		                      *FlatVector::IncrementalSelectionVector(), nullptr);
		const auto hash_values = hashes.Values<hash_t>();
		partition_indices.SetVectorType(VectorType::FLAT_VECTOR);
		auto writer = FlatVector::Writer<hash_t>(partition_indices, count);
		for (idx_t i = 0; i < count; i++) {
			writer.WriteValue((hash_values[i].GetValueUnsafe() >> shift) & mask);
		}
	}

	idx_t MaxPartitionIndex() const override {
		return mask;
	}

private:
	const idx_t shift;
	const hash_t mask;
	const idx_t hash_col_idx;
};

BoundedPartitionNode::BoundedPartitionNode(idx_t radix_bits_p, idx_t partition_idx_p)
    : radix_bits(radix_bits_p), partition_idx(partition_idx_p) {
}

RadixHTBoundedPartitioning::RadixHTBoundedPartitioning(RadixHTGlobalSinkState &gstate_p, idx_t radix_bits_p)
    : radix_bits(radix_bits_p), gstate(gstate_p), context(gstate_p.context),
      layout_ptr(gstate_p.radix_ht.GetLayoutPtr()) {
	D_ASSERT(radix_bits >= MINIMUM_RADIX_BITS && radix_bits <= MAXIMUM_RADIX_BITS);
	const auto num_targets = RadixPartitioning::NumberOfPartitions(radix_bits);
	targets.reserve(num_targets);
	for (idx_t i = 0; i < num_targets; i++) {
		targets.push_back(make_uniq<BoundedPartitionTarget>());
		targets.back()->root = make_uniq<BoundedPartitionNode>(radix_bits, i);
	}
	const auto block_size = gstate.block_alloc_size;
	handover_bytes = block_size / 2;
	ring_bytes =
	    MinValue<idx_t>(MaxValue<idx_t>(gstate.memory_limit / (8 * num_targets), 2 * block_size), 16 * block_size);
	// Leaves must stay small enough that several threads can combine one after it spilled
	const auto leaf_bytes =
	    MinValue<idx_t>(MaxValue<idx_t>(gstate.memory_limit / (4 * num_targets), block_size), MAXIMUM_LEAF_BYTES);
	const auto entry_bytes = gstate.radix_ht.GetLayout().GetRowWidth() + 2 * sizeof(ht_entry_t);
	route_count = MaxValue<idx_t>(leaf_bytes / entry_bytes, 4 * STANDARD_VECTOR_SIZE);
	// Every leaf pins at least a block (and a heap block for variable-size data), so the memory limits the leaves
	const auto blocks_per_leaf = gstate.radix_ht.GetLayout().AllConstant() ? 1 : 2;
	maximum_leaf_count =
	    MaxValue<idx_t>(gstate.memory_limit / (LEAF_MEMORY_DIVISOR * blocks_per_leaf * block_size), num_targets);
	leaf_count = num_targets;
}

//! Destroy the aggregate states of rows that were never combined (the query did not finish)
static void DestroyCollectionStates(RowOperationsState &row_state, TupleDataLayout &layout, TupleDataCollection &data) {
	if (data.Count() == 0) {
		return;
	}
	TupleDataChunkIterator iterator(data, TupleDataPinProperties::DESTROY_AFTER_DONE, false);
	auto &row_locations = iterator.GetChunkState().row_locations;
	do {
		RowOperations::DestroyStates(row_state, layout, row_locations);
	} while (iterator.Next());
	data.Reset();
}

static void DestroyNodeStates(RowOperationsState &row_state, TupleDataLayout &layout, BoundedPartitionNode &node) {
	for (auto &pending : node.pending) {
		DestroyCollectionStates(row_state, layout, *pending);
	}
	if (node.spilled) {
		DestroyCollectionStates(row_state, layout, *node.spilled);
	}
	if (node.child_data) {
		node.child_data->FlushAppendState(*node.child_append_state);
		for (auto &partition : node.child_data->GetPartitions()) {
			DestroyCollectionStates(row_state, layout, *partition);
		}
	}
	for (auto &child : node.children) {
		DestroyNodeStates(row_state, layout, *child);
	}
}

RadixHTBoundedPartitioning::~RadixHTBoundedPartitioning() {
	// HTs destroy their own states, but the rows that wait for an HT hold states too
	// The operator (and its layout) may be gone already, so use the layout that was kept
	auto layout = layout_ptr->Copy();
	if (!layout.HasDestructor()) {
		return;
	}
	try {
		ArenaAllocator allocator(Allocator::DefaultAllocator());
		RowOperationsState row_state(allocator);
		for (auto &target : targets) {
			for (auto &data : target->queue) {
				DestroyCollectionStates(row_state, layout, *data);
			}
			DestroyNodeStates(row_state, layout, *target->root);
		}
	} catch (...) { // NOLINT: destroying the states pins their blocks, which can fail after an out-of-memory error
	}
}

bool RadixHTBoundedPartitioning::CanHandOver(RadixHTLocalSinkState &lstate) const {
	if (!lstate.ht) {
		return false;
	}
	const auto partition_count = lstate.ht->GetPartitionedData().PartitionCount();
	return RadixPartitioning::RadixBitsOfPowerOfTwo(partition_count) >= radix_bits &&
	       lstate.ht->GetRadixBits() >= MINIMUM_RADIX_BITS;
}

bool RadixHTBoundedPartitioning::IsOwner(idx_t target_idx, idx_t thread_idx) const {
	return target_idx % gstate.number_of_threads == thread_idx % gstate.number_of_threads;
}

void RadixHTBoundedPartitioning::HandOver(RadixHTLocalSinkState &lstate, bool flush) {
	D_ASSERT(CanHandOver(lstate));
	auto &ht = *lstate.ht;
	const auto &partitioned_data = ht.GetPartitionedData();
	const auto partition_count = partitioned_data.PartitionCount();
	const auto shift = RadixPartitioning::RadixBitsOfPowerOfTwo(partition_count) - radix_bits;
	for (idx_t partition_idx = 0; partition_idx < partition_count; partition_idx++) {
		const auto &partition = partitioned_data.GetPartition(partition_idx);
		if (partition.Count() == 0 || (!flush && partition.DataSizeInBytes() < handover_bytes)) {
			continue;
		}
		Push(partition_idx >> shift, ht.AcquirePartition(partition_idx), lstate.thread_idx, true);
	}
	PollOwned(lstate.thread_idx);
}

void RadixHTBoundedPartitioning::Combine(RadixHTLocalSinkState &lstate) {
	D_ASSERT(CanHandOver(lstate));
	auto &ht = *lstate.ht;
	auto partitioned_data = ht.AcquirePartitionedData();
	ht.Abandon();
	const auto partition_count = partitioned_data->PartitionCount();
	const auto shift = RadixPartitioning::RadixBitsOfPowerOfTwo(partition_count) - radix_bits;
	for (idx_t partition_idx = 0; partition_idx < partition_count; partition_idx++) {
		if (partitioned_data->GetPartition(partition_idx).Count() == 0) {
			continue;
		}
		Push(partition_idx >> shift, partitioned_data->AcquirePartition(partition_idx), lstate.thread_idx, false);
	}

	// Help draining the targets that are not drained by another thread
	const auto num_targets = targets.size();
	for (idx_t i = 0; i < num_targets; i++) {
		auto &target = *targets[(lstate.thread_idx + i) % num_targets];
		if (target.queued_bytes.load() != 0) {
			Drain(target, false);
		}
	}
}

void RadixHTBoundedPartitioning::Push(idx_t target_idx, unique_ptr<TupleDataCollection> data, idx_t thread_idx,
                                      bool wait) {
	auto &target = *targets[target_idx];
	// Queued data is not pinned, but it holds whole blocks
	const auto bytes = data->GetBlockAllocationSize();
	idx_t queued;
	{
		lock_guard<mutex> guard(target.queue_lock);
		target.queue.push_back(std::move(data));
		queued = target.queued_bytes.fetch_add(bytes) + bytes;
	}
	if (queued >= ring_bytes) {
		// The ring is full: wait for the target instead of queueing more
		Drain(target, wait);
	} else if (queued >= ring_bytes / 2 && (IsOwner(target_idx, thread_idx) || queued >= ring_bytes / 4 * 3)) {
		Drain(target, false);
	}
}

void RadixHTBoundedPartitioning::PollOwned(idx_t thread_idx) {
	for (idx_t target_idx = thread_idx % gstate.number_of_threads; target_idx < targets.size();
	     target_idx += gstate.number_of_threads) {
		auto &target = *targets[target_idx];
		if (target.queued_bytes.load(std::memory_order_relaxed) >= ring_bytes / 2) {
			Drain(target, false);
		}
	}
}

void RadixHTBoundedPartitioning::Drain(BoundedPartitionTarget &target, bool block) {
	// Whoever holds the lock re-checks the queue after releasing it, so data queued meanwhile is never stranded
	do {
		unique_lock<mutex> guard(target.tree_lock, std::defer_lock);
		if (block) {
			guard.lock();
		} else if (!guard.try_lock()) {
			return;
		}
		while (target.queued_bytes.load() != 0) {
			DrainLocked(target);
		}
		block = false;
	} while (target.queued_bytes.load() != 0);
}

void RadixHTBoundedPartitioning::DrainLocked(BoundedPartitionTarget &target) {
	vector<unique_ptr<TupleDataCollection>> batch;
	{
		lock_guard<mutex> guard(target.queue_lock);
		batch.swap(target.queue);
		target.queued_bytes = 0;
	}
	auto &root = *target.root;
	if (!root.IsRouter() && !root.pass_through && !memory_exceeded.load(std::memory_order_relaxed)) {
		// Size the HT for the whole batch at once
		idx_t batch_count = 0;
		for (auto &data : batch) {
			batch_count += data->Count();
		}
		ReserveLeaf(root, batch_count);
	}
	for (auto &data : batch) {
		Absorb(*target.root, *data);
		data.reset();
	}
}

void RadixHTBoundedPartitioning::Absorb(BoundedPartitionNode &node, TupleDataCollection &data) {
	if (data.Count() == 0) {
		return;
	}
	if (node.IsRouter()) {
		auto &child_data = *node.child_data;
		TupleDataChunkIterator iterator(data, TupleDataPinProperties::DESTROY_AFTER_DONE, true);
		auto &chunk_state = iterator.GetChunkState();
		do {
			context.InterruptCheck();
			child_data.Append(*node.child_append_state, chunk_state, iterator.GetCurrentChunkCount());
		} while (iterator.Next());
		data.Reset();
		AbsorbChildren(node);
		return;
	}

	if (node.pass_through || memory_exceeded.load(std::memory_order_relaxed)) {
		PassThrough(node, data);
		return;
	}
	ReserveLeaf(node, data.Count());
	node.absorbed_count += data.Count();
	node.ht->Combine(data);
	Account(node);

	// Spilled groups count too (with their duplicates): they are combined into a single HT eventually
	const auto groups = node.ht->Count() + (node.spilled ? node.spilled->Count() : 0);
	if (groups > route_count && node.radix_bits + MINIMUM_FANOUT_BITS <= MAXIMUM_RADIX_BITS && Route(node)) {
		return;
	}
	if (OverBudget()) {
		Spill(node);
	}
}

void RadixHTBoundedPartitioning::PassThrough(BoundedPartitionNode &node, TupleDataCollection &data) {
	// Under memory pressure (or when aggregating does not pay off), a leaf keeps nothing pinned: its rows are
	// combined when it is finalized
	if (node.ht) {
		Spill(node);
		if (node.ht) {
			KeepAllocator(*node.ht);
			node.ht.reset();
			Account(node);
		}
	}
	node.pass_through = true;
	if (!node.spilled) {
		node.spilled = make_uniq<TupleDataCollection>(BufferManager::GetBufferManager(context), layout_ptr,
		                                              MemoryTag::HASH_TABLE, nullptr, context);
	}
	node.spilled->Combine(data);
	// Keep the partitions small enough to be combined one at a time
	if (node.spilled->Count() > PASS_THROUGH_ROUTE_FACTOR * route_count &&
	    node.radix_bits + MINIMUM_FANOUT_BITS <= MAXIMUM_RADIX_BITS) {
		Route(node);
	}
}

void RadixHTBoundedPartitioning::ReserveLeaf(BoundedPartitionNode &node, idx_t incoming) {
	D_ASSERT(!node.IsRouter());
	if (!node.ht) {
		const auto capacity = GroupedAggregateHashTable::GetCapacityForCount(MinValue(incoming, route_count));
		node.ht = gstate.radix_ht.CreateHT(context, capacity, 0);
		return;
	}
	auto &ht = *node.ht;
	if (ht.Count() + incoming <= ht.ResizeThreshold() || ht.Count() >= route_count) {
		return;
	}
	// Grow by more than the HT would by itself, re-inserting the groups less often
	const auto count = MinValue((ht.Count() + incoming) * GROWTH_FACTOR, route_count);
	ht.Resize(MaxValue(GroupedAggregateHashTable::GetCapacityForCount(count), ht.Capacity() * 2));
}

void RadixHTBoundedPartitioning::AbsorbChildren(BoundedPartitionNode &node) {
	D_ASSERT(node.IsRouter());
	auto &child_data = *node.child_data;
	bool flushed = false;
	for (idx_t child_idx = 0; child_idx < node.children.size(); child_idx++) {
		const auto &partition = child_data.GetPartition(child_idx);
		if (partition.DataSizeInBytes() < gstate.block_alloc_size) {
			continue;
		}
		if (!flushed) {
			child_data.FlushAppendState(*node.child_append_state);
			flushed = true;
		}
		auto data = child_data.AcquirePartition(child_idx);
		Absorb(*node.children[child_idx], *data);
	}
}

//! The number of distinct groups that yields 'groups' distinct groups among 'seen' uniformly drawn rows
static double ExpectedGroups(idx_t seen, idx_t groups) {
	double lo = static_cast<double>(groups);
	double hi = 1e15;
	for (idx_t i = 0; i < 60; i++) {
		const double d = (lo + hi) / 2;
		if (d * (1 - std::exp(-static_cast<double>(seen) / d)) < static_cast<double>(groups)) {
			lo = d;
		} else {
			hi = d;
		}
	}
	return hi;
}

bool RadixHTBoundedPartitioning::Route(BoundedPartitionNode &node) {
	D_ASSERT(!node.IsRouter());
	// Size the fan-out such that the children fit the groups that are expected
	auto fanout_bits = MAXIMUM_FANOUT_BITS;
	const auto groups = node.ht ? node.ht->Count() : 0;
	if (node.ht && node.absorbed_count > groups) {
		const auto expected = ExpectedGroups(node.absorbed_count, groups) / static_cast<double>(route_count / 2);
		const auto children = NextPowerOfTwo(LossyNumericCast<idx_t>(MinValue(expected, 1e6)) + 1);
		fanout_bits = MinValue(MaxValue(RadixPartitioning::RadixBitsOfPowerOfTwo(children), MINIMUM_FANOUT_BITS),
		                       MAXIMUM_FANOUT_BITS);
	}
	fanout_bits = MinValue(fanout_bits, MAXIMUM_RADIX_BITS - node.radix_bits);
	// Replacing this leaf by its children must not exceed the maximum number of leaves
	while (true) {
		const auto added = RadixPartitioning::NumberOfPartitions(fanout_bits) - 1;
		auto current = leaf_count.load();
		if (current + added > maximum_leaf_count) {
			if (fanout_bits == 1) {
				return false;
			}
			fanout_bits--;
			continue;
		}
		if (leaf_count.compare_exchange_weak(current, current + added)) {
			break;
		}
	}

	node.child_bits = fanout_bits;
	const auto child_radix_bits = node.radix_bits + fanout_bits;
	const auto child_count = RadixPartitioning::NumberOfPartitions(fanout_bits);
	for (idx_t child_idx = 0; child_idx < child_count; child_idx++) {
		node.children.push_back(
		    make_uniq<BoundedPartitionNode>(child_radix_bits, (node.partition_idx << fanout_bits) | child_idx));
	}

	for (auto &child : node.children) {
		child->pass_through = node.pass_through;
	}
	// Move the groups (and the spilled groups) of this node into the children
	unique_ptr<TupleDataCollection> data;
	if (node.ht) {
		data = node.ht->AcquirePartitionedData()->GetUnpartitioned();
		KeepAllocator(*node.ht);
		node.ht.reset();
		Account(node);
	}
	if (node.spilled) {
		auto spilled = std::move(node.spilled);
		auto parts = Split(*spilled, node.radix_bits, fanout_bits);
		for (idx_t child_idx = 0; child_idx < child_count; child_idx++) {
			if (parts[child_idx]->Count() != 0) {
				node.children[child_idx]->spilled = std::move(parts[child_idx]);
			}
		}
	}
	node.child_data = make_uniq<BoundedChildPartitionedTupleData>(BufferManager::GetBufferManager(context), layout_ptr,
	                                                              node.radix_bits, fanout_bits, context);
	node.child_append_state = make_uniq<PartitionedTupleDataAppendState>();
	node.child_data->InitializeAppendState(*node.child_append_state);
	if (data) {
		Absorb(node, *data);
	}
	return true;
}

void RadixHTBoundedPartitioning::Spill(BoundedPartitionNode &node) {
	if (!node.ht || node.ht->Count() == 0) {
		return;
	}
	// When most absorbed rows were new groups, the HT did not reduce the data enough to be worth spilling again
	node.pass_through =
	    static_cast<double>(node.ht->Count()) > PASS_THROUGH_THRESHOLD * static_cast<double>(node.absorbed_count);
	// The HT is discarded rather than cleared, a fresh one is created when rows arrive again
	auto data = node.ht->AcquirePartitionedData()->GetUnpartitioned();
	KeepAllocator(*node.ht);
	node.ht.reset();
	if (node.spilled) {
		node.spilled->Combine(*data);
	} else {
		node.spilled = std::move(data);
	}
	node.absorbed_count = 0;
	Account(node);
}

void RadixHTBoundedPartitioning::Account(BoundedPartitionNode &node) {
	idx_t size = 0;
	if (node.ht) {
		// The arena of the aggregate states cannot be spilled, leave it out
		size = node.ht->GetSizeInBytes() - node.ht->GetAggregateAllocator()->AllocationSize();
	}
	if (size >= node.accounted_size) {
		pinned_size.fetch_add(size - node.accounted_size, std::memory_order_relaxed);
	} else {
		pinned_size.fetch_sub(node.accounted_size - size, std::memory_order_relaxed);
	}
	node.accounted_size = size;
}

bool RadixHTBoundedPartitioning::OverBudget() {
	const auto budget = [&]() {
		// The local HTs of the threads need the rest
		return gstate.temporary_memory_state->GetReservation() / BUDGET_DIVISOR;
	};
	const auto pinned = pinned_size.load(std::memory_order_relaxed);
	if (pinned <= budget()) {
		return false;
	}
	if (memory_exceeded.load(std::memory_order_relaxed)) {
		return true;
	}
	{
		// Ask for more memory before evicting
		const annotated_lock_guard<annotated_mutex> guard {gstate.lock};
		auto &temporary_memory_state = *gstate.temporary_memory_state;
		// The leaf HTs get a fraction of the reservation, ask for room to double them
		const auto request = BUDGET_DIVISOR * 2 * pinned + gstate.minimum_reservation;
		if (temporary_memory_state.GetRemainingSize() < request) {
			temporary_memory_state.SetRemainingSizeAndUpdateReservation(context.GetContext(), request);
		}
	}
	if (pinned <= budget()) {
		return false;
	}
	// The groups do not fit: from now on, leaves keep nothing pinned
	memory_exceeded = true;
	return true;
}

vector<unique_ptr<TupleDataCollection>> RadixHTBoundedPartitioning::Split(TupleDataCollection &data,
                                                                          idx_t node_radix_bits, idx_t child_bits) {
	BoundedChildPartitionedTupleData split(BufferManager::GetBufferManager(context), layout_ptr, node_radix_bits,
	                                       child_bits, context);
	if (data.Count() != 0) {
		PartitionedTupleDataAppendState append_state;
		split.InitializeAppendState(append_state);
		TupleDataChunkIterator iterator(data, TupleDataPinProperties::DESTROY_AFTER_DONE, true);
		auto &chunk_state = iterator.GetChunkState();
		do {
			context.InterruptCheck();
			split.Append(append_state, chunk_state, iterator.GetCurrentChunkCount());
		} while (iterator.Next());
		split.FlushAppendState(append_state);
		data.Reset();
	}
	vector<unique_ptr<TupleDataCollection>> result;
	for (idx_t child_idx = 0; child_idx < split.PartitionCount(); child_idx++) {
		result.push_back(split.AcquirePartition(child_idx));
	}
	return result;
}

void RadixHTBoundedPartitioning::KeepAllocator(GroupedAggregateHashTable &ht) {
	auto allocator = ht.GetAggregateAllocator();
	if (allocator->AllocationSize() == 0) {
		return;
	}
	lock_guard<mutex> guard(allocator_lock);
	allocators.push_back(std::move(allocator));
}

void RadixHTBoundedPartitioning::PrepareNode(BoundedPartitionNode &node) {
	if (!node.IsRouter()) {
		leaves.push_back(node);
		return;
	}
	// The rows that wait for the children are absorbed when the children are finalized
	node.child_data->FlushAppendState(*node.child_append_state);
	for (idx_t child_idx = 0; child_idx < node.children.size(); child_idx++) {
		auto &child = *node.children[child_idx];
		auto data = node.child_data->AcquirePartition(child_idx);
		if (data->Count() != 0) {
			child.pending.push_back(std::move(data));
		}
		if (child.IsRouter()) {
			// Pending rows of a router belong to its children
			for (auto &pending : child.pending) {
				auto parts = Split(*pending, child.radix_bits, child.child_bits);
				for (idx_t part_idx = 0; part_idx < parts.size(); part_idx++) {
					if (parts[part_idx]->Count() != 0) {
						child.children[part_idx]->pending.push_back(std::move(parts[part_idx]));
					}
				}
			}
			child.pending.clear();
		}
		PrepareNode(child);
	}
}

void RadixHTBoundedPartitioning::PrepareLeaves() {
	D_ASSERT(leaves.empty());
	for (auto &target : targets) {
		lock_guard<mutex> guard(target->tree_lock);
		DrainLocked(*target);
		PrepareNode(*target->root);
	}
}

idx_t RadixHTBoundedPartitioning::LeafCount() const {
	return leaves.size();
}

void RadixHTBoundedPartitioning::GetLeafSize(idx_t leaf_idx, idx_t &count, idx_t &size, bool &combined) {
	auto &node = leaves[leaf_idx].get();
	count = 0;
	size = 0;
	if (node.ht) {
		count += node.ht->GetMaterializedCount();
		size += node.ht->GetPartitionedData().SizeInBytes();
	}
	if (node.spilled) {
		count += node.spilled->Count();
		size += node.spilled->SizeInBytes();
	}
	for (auto &pending : node.pending) {
		count += pending->Count();
		size += pending->SizeInBytes();
	}
	combined = !node.spilled && node.pending.empty();
}

unique_ptr<TupleDataCollection>
RadixHTBoundedPartitioning::FinalizeLeaf(idx_t leaf_idx, vector<shared_ptr<ArenaAllocator>> &allocators_out) {
	auto &node = leaves[leaf_idx].get();
	D_ASSERT(!node.IsRouter());
	const auto take_ht_data = [&](GroupedAggregateHashTable &ht) {
		auto data = ht.AcquirePartitionedData()->GetUnpartitioned();
		auto allocator = ht.GetAggregateAllocator();
		if (allocator->AllocationSize() != 0) {
			allocators_out.push_back(std::move(allocator));
		}
		return data;
	};

	unique_ptr<TupleDataCollection> result;
	if (!node.spilled) {
		// Absorb the pending rows into the HT of the leaf
		if (!node.pending.empty()) {
			idx_t pending_count = 0;
			for (auto &pending : node.pending) {
				pending_count += pending->Count();
			}
			ReserveLeaf(node, pending_count);
			for (auto &pending : node.pending) {
				node.ht->Combine(*pending);
			}
			node.pending.clear();
		}
		result = node.ht ? take_ht_data(*node.ht) : nullptr;
	} else {
		// The groups may have duplicates: combine everything in a fresh HT
		auto data = std::move(node.spilled);
		if (node.ht) {
			data->Combine(*take_ht_data(*node.ht));
		}
		for (auto &pending : node.pending) {
			data->Combine(*pending);
		}
		node.pending.clear();
		const auto capacity = GroupedAggregateHashTable::GetCapacityForCount(MinValue(data->Count(), route_count));
		auto ht = gstate.radix_ht.CreateHT(context, capacity, 0);
		ht->Combine(*data);
		result = take_ht_data(*ht);
	}
	if (node.ht) {
		node.ht.reset();
		Account(node);
	}
	{
		lock_guard<mutex> guard(allocator_lock);
		for (auto &allocator : allocators) {
			allocators_out.push_back(std::move(allocator));
		}
		allocators.clear();
	}
	return result;
}

void RadixHTBoundedPartitioning::GetLeaves(vector<BoundedPartitionLeafData> &leaves_out,
                                           vector<shared_ptr<ArenaAllocator>> &allocators_out) {
	PrepareLeaves();
	for (auto &leaf : leaves) {
		auto &node = leaf.get();
		BoundedPartitionLeafData leaf_data;
		leaf_data.combined = !node.spilled && node.pending.empty();
		leaf_data.data = std::move(node.spilled);
		for (auto &pending : node.pending) {
			if (leaf_data.data) {
				leaf_data.data->Combine(*pending);
			} else {
				leaf_data.data = std::move(pending);
			}
		}
		node.pending.clear();
		if (node.ht) {
			auto data = node.ht->AcquirePartitionedData()->GetUnpartitioned();
			auto allocator = node.ht->GetAggregateAllocator();
			if (allocator->AllocationSize() != 0) {
				allocators_out.push_back(std::move(allocator));
			}
			node.ht.reset();
			Account(node);
			if (leaf_data.data) {
				leaf_data.data->Combine(*data);
			} else {
				leaf_data.data = std::move(data);
			}
		}
		if (leaf_data.data && leaf_data.data->Count() != 0) {
			leaves_out.push_back(std::move(leaf_data));
		}
	}
	lock_guard<mutex> guard(allocator_lock);
	for (auto &allocator : allocators) {
		allocators_out.push_back(std::move(allocator));
	}
	allocators.clear();
}

} // namespace duckdb
