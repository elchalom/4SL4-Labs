from kedro.pipeline import Pipeline, node, pipeline
from ...nodes import data as dn
from ...nodes import experiments as ex
from ...nodes import selection as sel

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=dn.make_datasets,
            inputs=dict(N="params:N", seed="params:seed", noise_var="params:noise_var"),
            outputs="datasets",
            name="make_datasets",
        ),
        node(
            func=ex.degree_sweep,
            inputs=dict(dsets="datasets", m_min="params:m_min", m_max="params:m_max"),
            outputs="degree_sweep_result",
            name="degree_sweep",
        ),
        node(
            func=ex.plot_degree_curves,
            inputs=dict(dsets="datasets", result="degree_sweep_result", out_png="params:plot_degree_curve_png"),
            outputs=None,
            name="plot_degree_curves",
        ),
        node(
            func=ex.plot_fit_for_m,
            inputs=dict(dsets="datasets", M="params:plot_M_example", out_png="params:plot_M_example_png"),
            outputs=None,
            name="plot_fit_for_chosen_M_example",
        ),
        node(
            func=ex.ridge_sweep,
            inputs=dict(dsets="datasets", M="params:M_full", lambdas="params:lambdas"),
            outputs="ridge_sweep_result",
            name="ridge_sweep",
        ),
        node(
            func=ex.plot_ridge_curves,
            inputs=dict(result="ridge_sweep_result", out_png="params:plot_ridge_curve_png"),
            outputs=None,
            name="plot_ridge_curves",
        ),
        node(
            func=ex.plot_ridge_fits,
            inputs=dict(
                dsets="datasets",
                M="params:M_full",
                lambdas_show="params:lambdas_show",
                out_pngs="params:ridge_fit_pngs"
            ),
            outputs=None,
            name="plot_ridge_fit_examples",
        ),
        node(
            func=sel.select_and_test_best,
            inputs=dict(
                dsets="datasets",
                degree_sweep_result="degree_sweep_result",
                ridge_sweep_result="ridge_sweep_result",
                M_full="params:M_full"
            ),
            outputs="best_model_summary",
            name="select_and_test_best",
        ),
    ])
