//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/radix_ht_bounded.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/atomic.hpp"
#include "duckdb/common/types/row/partitioned_tuple_data.hpp"

namespace duckdb {

class ArenaAllocator;
class ClientContext;
class GroupedAggregateHashTable;
class PartitionedTupleData;
class RadixHTGlobalSinkState;
class RadixHTLocalSinkState;

//! A node of a bounded partition: a leaf aggregates into a bounded HT, a router splits its input over children
struct BoundedPartitionNode {
	BoundedPartitionNode(idx_t radix_bits, idx_t partition_idx);

	//! Radix bits of the hash prefix this node covers, and the prefix itself
	const idx_t radix_bits;
	const idx_t partition_idx;

	//! Leaf: the HT that groups are aggregated into
	unique_ptr<GroupedAggregateHashTable> ht;
	//! Leaf: groups evicted from the HT under memory pressure, may contain duplicates
	unique_ptr<TupleDataCollection> spilled;
	//! Leaf: size of the HT that is accounted in the pinned size of the partitioning
	idx_t accounted_size = 0;
	//! Leaf: rows absorbed into the HT since it was created or last spilled
	idx_t absorbed_count = 0;
	//! Leaf: whether rows go to the spilled groups directly, because the HT does not reduce them enough
	bool pass_through = false;
	//! Leaf: rows that the parent had not passed on when the Sink finished
	vector<unique_ptr<TupleDataCollection>> pending;

	//! Router: the children (empty for a leaf)
	vector<unique_ptr<BoundedPartitionNode>> children;
	idx_t child_bits = 0;
	//! Router: rows that wait for the children, which absorb them in batches of whole blocks
	unique_ptr<PartitionedTupleData> child_data;
	unique_ptr<PartitionedTupleDataAppendState> child_append_state;

	bool IsRouter() const {
		return !children.empty();
	}
};

//! Level-1 target: producers queue partition data, one thread at a time drains it into the node tree
struct BoundedPartitionTarget {
	mutex queue_lock;
	vector<unique_ptr<TupleDataCollection>> queue;
	atomic<idx_t> queued_bytes {0};

	mutex tree_lock;
	unique_ptr<BoundedPartitionNode> root;
};

//! The data of one leaf after the Sink, ready to be scanned (combined) or to be combined again (not combined)
struct BoundedPartitionLeafData {
	unique_ptr<TupleDataCollection> data;
	bool combined;
};

//! Bounded partitioning: threads hand their (pre-aggregated) partitions to shared per-partition targets while
//! sinking. Targets aggregate their data into HTs of bounded size, routing into finer partitions when full, and
//! evicting HTs to spillable storage when memory runs short. Memory is bounded by the groups, not by the input.
class RadixHTBoundedPartitioning {
public:
	RadixHTBoundedPartitioning(RadixHTGlobalSinkState &gstate, idx_t radix_bits);
	~RadixHTBoundedPartitioning();

	//! Whether the local HT of this thread can hand over its data
	bool CanHandOver(RadixHTLocalSinkState &lstate) const;
	//! Hand over the partitions of the (abandoned) local HT that are large enough (all of them if flush)
	void HandOver(RadixHTLocalSinkState &lstate, bool flush);
	//! Hand over everything the local HT holds and drain the targets
	void Combine(RadixHTLocalSinkState &lstate);
	//! Size of the leaf HTs, which are pinned
	idx_t GetPinnedSize() const {
		return pinned_size.load(std::memory_order_relaxed);
	}
	//! Drain all remaining data, and move the data of the leaves out of the partitioning
	void GetLeaves(vector<BoundedPartitionLeafData> &leaves, vector<shared_ptr<ArenaAllocator>> &allocators);
	//! Drain all remaining data, and collect the leaves (after all threads have combined)
	void PrepareLeaves();
	idx_t LeafCount() const;
	//! The rows that a leaf holds, their size, and whether they are combined already
	void GetLeafSize(idx_t leaf_idx, idx_t &count, idx_t &size, bool &combined);
	//! Combine everything a leaf holds, and return its groups (each exactly once)
	unique_ptr<TupleDataCollection> FinalizeLeaf(idx_t leaf_idx, vector<shared_ptr<ArenaAllocator>> &allocators);

public:
	//! Radix bits of the level-1 targets
	const idx_t radix_bits;

private:
	void Push(idx_t target_idx, unique_ptr<TupleDataCollection> data, idx_t thread_idx, bool wait);
	bool IsOwner(idx_t target_idx, idx_t thread_idx) const;
	void PollOwned(idx_t thread_idx);
	void Drain(BoundedPartitionTarget &target, bool block);
	void DrainLocked(BoundedPartitionTarget &target);

	void Absorb(BoundedPartitionNode &node, TupleDataCollection &data);
	void AbsorbChildren(BoundedPartitionNode &node);
	void ReserveLeaf(BoundedPartitionNode &node, idx_t incoming);
	bool Route(BoundedPartitionNode &node);
	void Spill(BoundedPartitionNode &node);
	void PassThrough(BoundedPartitionNode &node, TupleDataCollection &data);
	void Account(BoundedPartitionNode &node);
	bool OverBudget();
	vector<unique_ptr<TupleDataCollection>> Split(TupleDataCollection &data, idx_t node_radix_bits, idx_t child_bits);
	void KeepAllocator(GroupedAggregateHashTable &ht);
	void PrepareNode(BoundedPartitionNode &node);

private:
	RadixHTGlobalSinkState &gstate;
	ClientContext &context;
	shared_ptr<TupleDataLayout> layout_ptr;
	unsafe_vector<unique_ptr<BoundedPartitionTarget>> targets;

	//! Bytes per partition that a thread collects before handing them over
	idx_t handover_bytes;
	//! Queued bytes per target at which the owner drains (half), helpers drain (3/4), and producers must drain (all)
	idx_t ring_bytes;
	//! Groups at which a leaf routes into children
	idx_t route_count;
	//! The number of leaves, and the maximum that fits in memory
	atomic<idx_t> leaf_count;
	idx_t maximum_leaf_count;

	//! Total size of the leaf HTs, which are pinned
	atomic<idx_t> pinned_size {0};
	//! Whether the leaf HTs exceeded the memory budget, after which leaves keep nothing pinned
	atomic<bool> memory_exceeded {false};

	//! The leaves, once all threads have combined
	vector<reference<BoundedPartitionNode>> leaves;

	//! Arenas of HTs that were discarded, states may still point into them
	mutex allocator_lock;
	vector<shared_ptr<ArenaAllocator>> allocators;

public:
	//! The maximum radix bits of a leaf
	static constexpr idx_t MAXIMUM_RADIX_BITS = 12;
	//! The minimum radix bits of the level-1 targets (local HTs need partitioned data that is not pinned)
	static constexpr idx_t MINIMUM_RADIX_BITS = 3;
	//! Bounded partitioning needs at least one target per this many threads
	static constexpr idx_t PARTITIONS_PER_THREAD_DIVISOR = 2;
	//! The leaf HTs may take this fraction of the reservation, the local HTs of the threads need the rest
	static constexpr idx_t BUDGET_DIVISOR = 4;
	//! The minimum blocks of all leaves together take at most this fraction of the memory limit
	static constexpr idx_t LEAF_MEMORY_DIVISOR = 16;
	//! Maximum leaf HT size (pointer table and rows) at which a leaf routes
	static constexpr idx_t MAXIMUM_LEAF_BYTES = 64ULL * 1024ULL * 1024ULL;
	//! A leaf that spills with more groups than this fraction of the rows it absorbed stops aggregating
	static constexpr double PASS_THROUGH_THRESHOLD = 0.5;
	//! A passing-through leaf routes once it holds this many times the groups at which a leaf routes
	static constexpr idx_t PASS_THROUGH_ROUTE_FACTOR = 4;
	//! Factor by which leaf HTs grow
	static constexpr idx_t GROWTH_FACTOR = 8;
	//! Minimum and maximum fan-out (in bits) when routing
	static constexpr idx_t MINIMUM_FANOUT_BITS = 2;
	static constexpr idx_t MAXIMUM_FANOUT_BITS = 4;
};

} // namespace duckdb
