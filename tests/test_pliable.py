"""Estimator, mathematical, and integration checks on small generated fixtures."""
import pickle

import numpy as np
from numpy.testing import assert_allclose
import pytest
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from structured_linear import PliableRegressor
from structured_linear._solver import InteractionDesign, penalty, proximal


@pytest.fixture
def problem():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(60, 5))
    y = 1.3 + data[:, 0] * (1.1 + .8 * data[:, 3]) - .6 * data[:, 1] + .3 * data[:, 4]
    y += rng.normal(scale=.1, size=len(y))
    return data, y


@parametrize_with_checks([PliableRegressor()])
def test_sklearn_estimator(estimator, check):
    check(estimator)


@pytest.mark.parametrize('intercept', [True, False])
@pytest.mark.parametrize('ratio', [0., .5, 1.])
@pytest.mark.parametrize('weighted', [True, False])
def test_convex_reference(problem, intercept, ratio, weighted):
    cp = pytest.importorskip('cvxpy')
    data, y = problem
    weight = np.linspace(.2, 2, len(y)) if weighted else np.ones(len(y))
    x, z = data[:, :3], data[:, 3:]
    beta = cp.Variable(3); gamma = cp.Variable(2); theta = cp.Variable((3, 2))
    b = cp.Variable() if intercept else 0.
    prediction = b + x @ beta + z @ gamma + cp.sum(cp.multiply(x, z @ theta.T), axis=1)
    reg = sum(cp.norm(cp.hstack([beta[j], theta[j, :]]), 2) + cp.norm(theta[j, :], 2) for j in range(3))
    objective = cp.sum(cp.multiply(weight, cp.square(y - prediction))) / (2 * weight.sum())
    objective += .08 * ((1 - ratio) * reg + ratio * cp.norm1(theta))
    reference = cp.Problem(cp.Minimize(objective))
    reference.solve(solver='CLARABEL', tol_gap_abs=1e-10, tol_feas=1e-10, tol_gap_rel=1e-10)
    model = PliableRegressor(alpha=.08, l1_ratio=ratio, modifier_features=[3, 4],
                         fit_intercept=intercept, tol=1e-9).fit(data, y, sample_weight=weight)
    assert model.converged_
    assert_allclose(model.objective_, reference.value, rtol=2e-7, atol=1e-9)
    assert_allclose(model.predict(data), prediction.value, atol=3e-5, rtol=1e-5)


@pytest.mark.parametrize('ratio', [0., .3, .8])
def test_no_modifiers_is_lasso(problem, ratio):
    data, y = problem
    model = PliableRegressor(alpha=.07, l1_ratio=ratio, tol=1e-10).fit(data, y)
    ref = Lasso(alpha=.07 * (1-ratio), tol=1e-11, max_iter=10000).fit(data, y)
    assert_allclose(model.predict(data), ref.predict(data), atol=1e-8)
    assert model.interaction_coef_.shape == (5, 0)


def test_proximal_reference():
    cp = pytest.importorskip('cvxpy')
    rng = np.random.default_rng(23); p, k = 3, 2
    for ratio in [0., .5, 1.]:
        v = rng.normal(size=p+k+p*k); u = cp.Variable(len(v))
        theta = cp.reshape(u[p+k:], (p, k), order='C')
        reg = sum(cp.norm(cp.hstack([u[j], theta[j, :]])) + cp.norm(theta[j, :]) for j in range(p))
        objective = .5 * cp.sum_squares(u-v) + .2*((1-ratio)*reg+ratio*cp.norm1(theta))
        ref = cp.Problem(cp.Minimize(objective)); ref.solve(solver='CLARABEL',tol_gap_abs=1e-10,tol_feas=1e-10,tol_gap_rel=1e-10)
        actual = proximal(v, p, k, 1., .2, ratio)
        assert_allclose(actual, u.value, atol=2e-5)
        assert_allclose(actual[p:p+k],v[p:p+k])


def test_matrix_free_design(problem):
    data, y = problem; x, z = data[:, :3], data[:, 3:]
    w = np.arange(1, len(y)+1); w = w/w.sum()
    raw = np.column_stack([x, z, (x[:, :, None]*z[:, None, :]).reshape(len(y), -1)])
    centered = raw - w @ raw
    design = InteractionDesign(x, z, w, True)
    v = np.arange(raw.shape[1]) / 7; residual = np.sin(y)
    assert_allclose(design.matvec(v), centered @ v, atol=1e-14)
    assert_allclose(design.rmatvec(residual), centered.T @ residual, atol=1e-13)


def test_weights_rescaling_replication_and_zero_rows(problem):
    data, y = problem; weights = np.resize([1, 2, 3], len(y))
    args = dict(alpha=.1, modifier_features=[3, 4], tol=1e-9)
    a = PliableRegressor(**args).fit(data, y, sample_weight=weights)
    b = PliableRegressor(**args).fit(data, y, sample_weight=weights*13)
    c = PliableRegressor(**args).fit(np.repeat(data,weights,axis=0), np.repeat(y,weights))
    assert_allclose(a.predict(data),b.predict(data),atol=1e-8)
    assert_allclose(a.predict(data),c.predict(data),atol=1e-7)
    extra = np.vstack([data, np.ones((3,5))*10]); ey = np.r_[y,[12,18,32]]
    d = PliableRegressor(**args).fit(extra,ey,sample_weight=np.r_[weights,[0,0,0]])
    assert_allclose(a.predict(data),d.predict(data),atol=1e-7)


def test_pipeline_search_names_clone_and_pickle(problem):
    pd = pytest.importorskip('pandas')
    data, y = problem; data = pd.DataFrame(data, columns=list('abcde'))
    estimator = PliableRegressor(modifier_features=[3,4],tol=1e-7)
    clone(estimator).fit(data,y)
    model = estimator.fit(data,y)
    assert list(model.feature_names_in_) == list('abcde')
    assert_allclose(pickle.loads(pickle.dumps(model)).predict(data),model.predict(data))
    pipe = make_pipeline(StandardScaler(),PliableRegressor(modifier_features=[3,4]))
    search = GridSearchCV(pipe,{'pliableregressor__alpha':[.01,.1]},cv=KFold(3),scoring='neg_mean_squared_error')
    search.fit(data,y)
    assert np.isfinite(search.best_score_)
    assert np.isfinite(search.predict(data)).all()


def test_refit_and_nonconvergence(problem):
    data, y = problem
    model = PliableRegressor(alpha=.01,modifier_features=[3,4],max_iter=1,tol=1e-14)
    with pytest.warns(ConvergenceWarning): model.fit(data,y)
    assert not model.converged_
    model.set_params(modifier_features=None,max_iter=10000,tol=1e-8).fit(data[:,:2],y)
    assert model.coef_.shape == (2,)
    assert model.interaction_coef_.shape == (2,0)
    with pytest.raises(NotFittedError): clone(model).predict(data[:,:2])


@pytest.mark.parametrize('params',[
    {'alpha':-1},{'alpha':np.nan},{'alpha':np.inf},{'l1_ratio':1.1},
    {'tol':0},{'max_iter':0},{'fit_intercept':'yes'},
    {'modifier_features':[3,3]},{'modifier_features':[-1]},
    {'modifier_features':[5]},{'modifier_features':[True,False]},
    {'modifier_features':[[1]]},{'modifier_features':[0,1,2,3,4]},
])
def test_invalid_parameters(problem,params):
    data,y=problem
    with pytest.raises((ValueError,TypeError)): PliableRegressor(**params).fit(data,y)


@pytest.mark.parametrize('weights',[np.zeros(60),np.full(60,-1),np.full(60,np.nan),np.ones((60,1)),np.ones(59)])
def test_invalid_weights(problem,weights):
    data,y=problem
    with pytest.raises(ValueError): PliableRegressor().fit(data,y,sample_weight=weights)


def test_weighted_metadata_routing(problem):
    from sklearn import config_context
    data, y = problem
    with config_context(enable_metadata_routing=True):
        scaler = StandardScaler().set_fit_request(sample_weight=True)
        model = PliableRegressor(alpha=.03, modifier_features=[3,4]).set_fit_request(sample_weight=True)
        pipeline = make_pipeline(scaler, model)
        weights = np.linspace(.1, 2., len(y))
        pipeline.fit(data,y,sample_weight=weights)
        assert np.isfinite(pipeline.predict(data)).all()


@pytest.mark.parametrize('constant', ['predictor','modifier','target','all'])
def test_constant_columns(problem,constant):
    data,y=problem;data=data.copy();y=y.copy()
    if constant=='predictor':data[:,0]=1.
    if constant=='modifier':data[:,3]=1.
    if constant=='target':y[:]=3.2
    if constant=='all':data[:]=1.;y[:]=3.2
    model=PliableRegressor(alpha=.1,modifier_features=[3,4],tol=1e-8).fit(data,y)
    assert model.converged_
    assert np.isfinite(model.objective_)
    if constant in ['target','all']:assert_allclose(model.predict(data),y,atol=1e-8)


def test_zero_penalty_and_redundant_modifiers(problem):
    data,y=problem;data=data.copy();data[:,4]=data[:,3]
    model=PliableRegressor(alpha=0,modifier_features=[3,4],tol=1e-9).fit(data,y)
    x,z=data[:,:3],data[:,3:]
    expanded=np.column_stack([np.ones(len(y)),data,(x[:,:,None]*z[:,None,:]).reshape(len(y),-1)])
    ref=expanded@np.linalg.lstsq(expanded,y,rcond=None)[0]
    assert_allclose(model.predict(data),ref,atol=1e-7)


def test_modifier_permutation(problem):
    data,y=problem
    a=PliableRegressor(alpha=.02,modifier_features=[3,4],tol=1e-9).fit(data,y)
    b=PliableRegressor(alpha=.02,modifier_features=[4,3],tol=1e-9).fit(data,y)
    assert_allclose(a.predict(data),b.predict(data),atol=1e-8)


def test_inputs_are_not_modified(problem):
    data,y=problem;original=data.copy();target=y.copy();weights=np.ones(len(y))
    PliableRegressor(alpha=.02,modifier_features=[3,4]).fit(data,y,sample_weight=weights)
    assert_allclose(data,original);assert_allclose(y,target);assert_allclose(weights,1.)


@pytest.mark.parametrize('change', ['target', 'rows', 'weights', 'ratio'])
def test_warm_start_matches_cold_fit(problem, change):
    data, target = problem
    model = PliableRegressor(alpha=.06, modifier_features=[3, 4], warm_start=True, tol=1e-9)
    model.fit(data, target)
    model.set_params(alpha=.03)
    weights = None
    if change == 'target':
        target = target + .2 * data[:, 1]
    if change == 'rows':
        data, target = data[5:], target[5:]
    if change == 'weights':
        weights = np.linspace(.5, 2., len(target))
    if change == 'ratio':
        model.set_params(l1_ratio=.8)
    cold = clone(model).set_params(warm_start=False).fit(data, target, sample_weight=weights)
    model.fit(data, target, sample_weight=weights)
    assert model.converged_ and cold.converged_
    assert_allclose(model.objective_, cold.objective_, atol=1e-10)
    assert_allclose(model.predict(data), cold.predict(data), atol=1e-7)


def test_warm_start_resets_on_schema_change(problem):
    data, target = problem
    model = PliableRegressor(alpha=.03, modifier_features=[3, 4], warm_start=True)
    model.fit(data, target)
    model.set_params(modifier_features=[4, 3])
    model.fit(data, target)
    cold = clone(model).fit(data, target)
    assert model.n_iter_ == cold.n_iter_
    assert_allclose(model.predict(data), cold.predict(data))
    model.set_params(warm_start=False)
    model.fit(data, target)
    assert model.n_iter_ == cold.n_iter_


def test_failed_refit_preserves_fitted_schema(problem):
    data, target = problem
    model = PliableRegressor(modifier_features=[3, 4], warm_start=True).fit(data, target)
    prediction = model.predict(data)
    with pytest.raises(ValueError, match='distinct indices'):
        model.fit(data[:, :4], target)
    assert_allclose(model.predict(data), prediction)
