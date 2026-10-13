// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 The Eigen Authors.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

#include <Eigen/Tensor>

using Eigen::DefaultDevice;
using Eigen::Tensor;
using Eigen::TensorEvaluator;

// Regression: TensorCwiseNullaryOp must report bytes_loaded == 0.
// NullaryOps (constants, Zero, Identity, Random, sequence generators)
// produce values from registers or minimal state without loading from
// memory. If they reported nonzero bytes_loaded, expressions dominated
// by constants (e.g. Horner-form polynomials) would be misclassified as
// memory-bound and the threadpool cost model would over-restrict
// parallelism. See TensorEvaluator.h, the CwiseNullaryOp specialization
// of costPerCoeff().
template <typename Scalar>
static void test_nullary_zero_bytes_loaded() {
  Tensor<Scalar, 1> shape(/*size=*/16);
  auto zeros = shape.constant(Scalar(0));
  auto sevens = shape.constant(Scalar(7));

  using ZeroEval = TensorEvaluator<const decltype(zeros), DefaultDevice>;
  using ConstEval = TensorEvaluator<const decltype(sevens), DefaultDevice>;

  DefaultDevice device;
  ZeroEval zero_eval(zeros, device);
  ConstEval const_eval(sevens, device);

  for (bool vectorized : {false, true}) {
    const auto zero_cost = zero_eval.costPerCoeff(vectorized);
    const auto const_cost = const_eval.costPerCoeff(vectorized);
    VERIFY_IS_EQUAL(zero_cost.bytes_loaded(), 0.0);
    VERIFY_IS_EQUAL(zero_cost.bytes_stored(), 0.0);
    VERIFY_IS_EQUAL(const_cost.bytes_loaded(), 0.0);
    VERIFY_IS_EQUAL(const_cost.bytes_stored(), 0.0);
  }
}

template <typename Scalar>
static void test_unary_binary_ternary_block_resource_cost() {
  Tensor<Scalar, 2> a(64, 64);
  Tensor<Scalar, 2> b(64, 64);
  Tensor<Scalar, 2> c(64, 64);
  a.setRandom();
  b.setRandom();
  c.setRandom();

  DefaultDevice device;

  auto unary_expr = -a;
  using UnaryEval = TensorEvaluator<const decltype(unary_expr), DefaultDevice>;
  UnaryEval unary_eval(unary_expr, device);
  if (UnaryEval::PacketAccess &&
      Eigen::internal::functor_traits<Eigen::internal::scalar_opposite_op<Scalar>>::Cost > 0) {
    VERIFY(unary_eval.getResourceRequirements().cost_per_coeff.compute_cycles() > 0.0);
  }

  auto binary_expr = a + b;
  using BinaryEval = TensorEvaluator<const decltype(binary_expr), DefaultDevice>;
  BinaryEval binary_eval(binary_expr, device);
  if (BinaryEval::PacketAccess &&
      Eigen::internal::functor_traits<Eigen::internal::scalar_sum_op<Scalar>>::Cost > 0) {
    VERIFY(binary_eval.getResourceRequirements().cost_per_coeff.compute_cycles() > 0.0);
  }
}

template <typename Scalar>
static void test_shuffling_cost_and_block_requirements() {
  Tensor<Scalar, 2> a(256, 256);
  a.setRandom();

  Eigen::array<Eigen::Index, 2> perm = {1, 0};
  auto shuffle_expr = a.shuffle(perm);
  using ShuffleEval = TensorEvaluator<const decltype(shuffle_expr), DefaultDevice>;

  DefaultDevice device;
  ShuffleEval eval(shuffle_expr, device);

  const auto scalar_cost = eval.costPerCoeff(/*vectorized=*/false);
  const auto vec_cost = eval.costPerCoeff(/*vectorized=*/true);
  if (ShuffleEval::PacketAccess && ShuffleEval::PacketSize > 1) {
    VERIFY(vec_cost.compute_cycles() < scalar_cost.compute_cycles());
  }

  const auto req = eval.getResourceRequirements();
  VERIFY(req.shape_type == Eigen::internal::TensorBlockShapeType::kUniformAllDims);
  // Inner-dimension shuffles should not artificially inflate block cost to 56 cycles/coeff.
  VERIFY(req.cost_per_coeff.compute_cycles() <= 2.0);
  // Small 256x256 2D transpose should stay single-threaded rather than paying
  // thread-pool dispatch and barrier overhead.
  const int threads_256 =
      Eigen::TensorCostModel<DefaultDevice>::numThreads(256.0 * 256.0, req.cost_per_coeff, /*max_threads=*/8);
  VERIFY_IS_EQUAL(threads_256, 1);

  // Larger 1024x1024 2D transpose should parallelize across threads.
  const int threads_1024 =
      Eigen::TensorCostModel<DefaultDevice>::numThreads(1024.0 * 1024.0, req.cost_per_coeff, /*max_threads=*/8);
  VERIFY(threads_1024 > 1);
}

EIGEN_DECLARE_TEST(tensor_cost_model) {
  CALL_SUBTEST(test_nullary_zero_bytes_loaded<float>());
  CALL_SUBTEST(test_nullary_zero_bytes_loaded<double>());
  CALL_SUBTEST(test_nullary_zero_bytes_loaded<int>());
  CALL_SUBTEST(test_unary_binary_ternary_block_resource_cost<float>());
  CALL_SUBTEST(test_unary_binary_ternary_block_resource_cost<double>());
  CALL_SUBTEST(test_shuffling_cost_and_block_requirements<float>());
  CALL_SUBTEST(test_shuffling_cost_and_block_requirements<double>());
}
