import alphashape
import matplotlib.pyplot as plt
import pandas as pd
from descartes import PolygonPatch
from matplotlib.cm import ScalarMappable

from dynast.analytics.visualize import frontier_builder  # , #plot_correlation


def plot_search_progression(
    results_path: str,
    evals_limit: int = None,
    random_results_path: str = None,
) -> None:
    df = pd.read_csv(results_path)

    if evals_limit:
        df = df[:evals_limit]

    df.columns = ['config', 'date', 'params', 'latency', 'macs', 'accuracy_top1']

    fig, ax = plt.subplots(figsize=(7, 5))

    cm = plt.cm.get_cmap('viridis_r')
    count = [x for x in range(len(df))]

    if random_results_path:
        df_random = pd.read_csv(random_results_path)
        df_random.columns = ['config', 'date', 'params', 'latency', 'macs', 'accuracy_top1']
        ax.scatter(
            df_random['macs'].values,
            df_random['accuracy_top1'].values,
            marker='.',
            alpha=0.1,
            c='grey',
            label='Random DNN Model',
        )
        cloud = list(df_random[['macs', 'accuracy_top1']].to_records(index=False))
        for i in range(len(cloud)):
            cloud[i] = list(cloud[i])
        alpha_shape = alphashape.alphashape(cloud, 0)
        ax.add_patch(
            PolygonPatch(
                alpha_shape,
                fill=None,
                alpha=0.4,
                color='grey',
                linewidth=1.5,
                label='Random search boundary',
                linestyle='--',
            )
        )

    ax.scatter(
        df['macs'].values,
        df['accuracy_top1'].values,
        marker='^',
        alpha=0.7,
        c=count,
        cmap=cm,
        label='Discovered DNN Model',
        s=10,
    )

    ax.set_title('DyNAS-T Search Results \n{}'.format(results_path.split('.')[0]))
    ax.set_xlabel('MACs', fontsize=13)
    ax.set_ylabel('Top1', fontsize=13)
    ax.legend(fancybox=True, fontsize=10, framealpha=1, borderpad=0.2, loc='lower right')
    ax.grid(True, alpha=0.3)
    # ax.set_ylim(72,77.5)

    df_conc_front = frontier_builder(df, optimization_metrics=['macs', 'accuracy_top1'])
    ax.plot(
        df_conc_front['macs'],
        df_conc_front['accuracy_top1'],
        color='red',
        linestyle='--',
        label='DyNAS-T Pareto front',
    )

    # Eval Count bar
    norm = plt.Normalize(0, len(df))
    sm = ScalarMappable(norm=norm, cmap=cm)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.85)
    cbar.ax.set_title("         Evaluation\n  Count", fontsize=8)

    fig.tight_layout(pad=2)
    plt.savefig('{}.png'.format(results_path.split('.')[0]))


# def correlation() -> None:
#     df_1 = pd.read_csv('bnas_1_random.csv')[:50]
#     df_2 = pd.read_csv('bnas_2_random.csv')[:50]
#     df_1.columns = ['config', 'date', 'params', 'latency', 'macs', 'accuracy_top1']
#     df_2.columns = ['config', 'date', 'params', 'latency', 'macs', 'accuracy_top1']

#     plot_correlation(x1=df_1['accuracy_top1'], x2=df_2['accuracy_top1'])


if __name__ == '__main__':
    plot_search_progression(
        # results_path='bnas_1_latency.csv',
        # results_path='results_tlt_linas_dist2_long.csv',
        # results_path='results_ofambv3_random_long.csv',
        results_path='bootstrapnas_resnet50_cifar10_linas.csv',
        # random_results_path='test.csv',
        # random_results_path='bnas_2_random.csv',
        # random_results_path='bnas_mbv2_cifar_top1_macs_random.csv',
    )

    plot_search_progression(
        results_path='bootstrapnas_resnet50_cifar10_random.csv',
    )

    # correlation()
