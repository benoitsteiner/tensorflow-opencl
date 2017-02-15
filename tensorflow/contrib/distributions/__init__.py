# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Classes representing statistical distributions and ops for working with them.

See the @{$python/contrib.distributions} guide.

@@ReparameterizationType
@@Distribution

@@Binomial
@@Bernoulli
@@BernoulliWithSigmoidProbs
@@Beta
@@BetaWithSoftplusConcentration
@@Categorical
@@Chi2
@@Chi2WithAbsDf
@@Exponential
@@ExponentialWithSoftplusRate
@@Gamma
@@GammaWithSoftplusConcentrationRate
@@InverseGamma
@@InverseGammaWithSoftplusConcentrationRate
@@Laplace
@@LaplaceWithSoftplusScale
@@Logistic
@@Normal
@@NormalWithSoftplusScale
@@Poisson
@@StudentT
@@StudentTWithAbsDfSoftplusScale
@@Uniform

@@MultivariateNormalDiag
@@MultivariateNormalTriL
@@MultivariateNormalDiagPlusLowRank
@@MultivariateNormalDiagWithSoftplusScale

@@Dirichlet
@@DirichletMultinomial
@@Multinomial
@@WishartCholesky
@@WishartFull

@@matrix_diag_transform

@@TransformedDistribution
@@QuantizedDistribution

@@Mixture

@@normal_conjugates_known_scale_posterior
@@normal_conjugates_known_scale_predictive

@@kl
@@RegisterKL

@@softplus_inverse

@@ExpRelaxedOneHotCategorical
@@OneHotCategorical
@@RelaxedBernoulli
@@RelaxedOneHotCategorical
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,wildcard-import,line-too-long,g-importing-member

from tensorflow.contrib.distributions.python.ops import bijector
from tensorflow.contrib.distributions.python.ops import conditional_bijector
from tensorflow.contrib.distributions.python.ops.bernoulli import *
from tensorflow.contrib.distributions.python.ops.beta import *
from tensorflow.contrib.distributions.python.ops.binomial import *
from tensorflow.contrib.distributions.python.ops.categorical import *
from tensorflow.contrib.distributions.python.ops.chi2 import *
from tensorflow.contrib.distributions.python.ops.conditional_distribution import *
from tensorflow.contrib.distributions.python.ops.conditional_transformed_distribution import *
from tensorflow.contrib.distributions.python.ops.dirichlet import *
from tensorflow.contrib.distributions.python.ops.dirichlet_multinomial import *
from tensorflow.contrib.distributions.python.ops.distribution import *
from tensorflow.contrib.distributions.python.ops.distribution_util import matrix_diag_transform
from tensorflow.contrib.distributions.python.ops.distribution_util import softplus_inverse
from tensorflow.contrib.distributions.python.ops.exponential import *
from tensorflow.contrib.distributions.python.ops.gamma import *
from tensorflow.contrib.distributions.python.ops.inverse_gamma import *
from tensorflow.contrib.distributions.python.ops.kullback_leibler import *
from tensorflow.contrib.distributions.python.ops.laplace import *
from tensorflow.contrib.distributions.python.ops.logistic import *
from tensorflow.contrib.distributions.python.ops.mixture import *
from tensorflow.contrib.distributions.python.ops.multinomial import *
from tensorflow.contrib.distributions.python.ops.mvn_diag import *
from tensorflow.contrib.distributions.python.ops.mvn_diag_plus_low_rank import *
from tensorflow.contrib.distributions.python.ops.mvn_tril import *
from tensorflow.contrib.distributions.python.ops.normal import *
from tensorflow.contrib.distributions.python.ops.normal_conjugate_posteriors import *
from tensorflow.contrib.distributions.python.ops.onehot_categorical import *
from tensorflow.contrib.distributions.python.ops.poisson import *
from tensorflow.contrib.distributions.python.ops.quantized_distribution import *
from tensorflow.contrib.distributions.python.ops.relaxed_bernoulli import *
from tensorflow.contrib.distributions.python.ops.relaxed_onehot_categorical import *
from tensorflow.contrib.distributions.python.ops.student_t import *
from tensorflow.contrib.distributions.python.ops.transformed_distribution import *
from tensorflow.contrib.distributions.python.ops.uniform import *
from tensorflow.contrib.distributions.python.ops.wishart import *

# pylint: enable=unused-import,wildcard-import,line-too-long,g-importing-member

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = ['bijector',
                    'ConditionalDistribution',
                    'ConditionalTransformedDistribution',
                    'FULLY_REPARAMETERIZED', 'NOT_REPARAMETERIZED']

remove_undocumented(__name__, _allowed_symbols)
