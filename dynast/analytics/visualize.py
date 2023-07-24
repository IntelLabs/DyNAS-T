# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import math
import os

import alphashape
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from descartes import PolygonPatch
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import ScalarFormatter
from pymoo.indicators.hv import HV
from scipy.spatial import Delaunay
from shapely.geometry import MultiLineString, MultiPoint, mapping
from shapely.ops import cascaded_union, polygonize
from sklearn.preprocessing import MinMaxScaler

from dynast.utils import log


PLOT_HYPERVOLUME = True
NORMALIZE = False

colors = {
    'blue': '#1f77b4',
    'orange': '#ff7f0e',
    'green': '#2ca02c',
    'red': '#d62728',
    'purple': '#9467bd',
    'brown': '#8c564b',
    'pink': '#e377c2',
    'gray': '#7f7f7f',
    'olive': '#bcbd22',
    'cyan': '#17becf',
}



def frontier_builder(df, optimization_metrics, alpha=0, verbose=False):
    """
    Modified alphashape algorithm to draw Pareto Front for OFA search.
    Takes a DataFrame of column form [x, y] = [latency, accuracy]

    Params:
    df     - dataframe containing `optimization_metrics` columns at minimum
    alpha  - Dictates amount of tolerable 'concave-ness' allowed.
             A fully convex front will be given if 0 (also better for runtime)
    """
    if verbose:
        log.info('Running front builder')
    df = df[optimization_metrics]
    points = list(df.to_records(index=False))
    for i in range(len(points)):
        points[i] = list(points[i])
    points = MultiPoint(points)

    # TODO(macsz) Fix the line below in comment:
    # if len(points) < 4 or alpha <= 0:
    if alpha <= 0:
        if verbose:
            log.info('Alpha=0 -> convex hull')
        result = points.convex_hull
    else:
        coords = np.array([point.coords[0] for point in points])
        tri = Delaunay(coords)
        edges = set()
        edge_points = []
        edge_out = []

        # Loop over triangles
        for ia, ib, ic in tri.vertices:
            pa = coords[ia]
            pb = coords[ib]
            pc = coords[ic]

            # Lengths of sides of triangle
            a = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
            b = math.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
            c = math.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)

            # Semiperimeter of triangle
            s = (a + b + c) * 0.5

            # Area of triangle by Heron's formula
            # Precompute value inside square root to avoid unbound math error in
            # case of 0 area triangles.
            area = s * (s - a) * (s - b) * (s - c)

            if area > 0:
                area = math.sqrt(area)

                # Radius Filter
                if a * b * c / (4.0 * area) < 1.0 / alpha:
                    for i, j in itertools.combinations([ia, ib, ic], r=2):
                        if (i, j) not in edges and (j, i) not in edges:
                            edges.add((i, j))
                            edge_points.append(coords[[i, j]])

                            if coords[i].tolist() not in edge_out:
                                edge_out.append(coords[i].tolist())
                            if coords[j].tolist() not in edge_out:
                                edge_out.append(coords[j].tolist())

        # Create the resulting polygon from the edge points
        m = MultiLineString(edge_points)
        triangles = list(polygonize(m))
        result = cascaded_union(triangles)

    # Find multi-polygon boundary
    bound = list(mapping(result.boundary)['coordinates'])

    # Cutoff non-Pareto front points
    # note that extreme concave geometries will create issues if bi-sected by line
    df = pd.DataFrame(bound, columns=['x', 'y'])

    # y=mx+b
    left_point = (df.iloc[df.idxmin()[0]][0], df.iloc[df.idxmin()[0]][1])
    right_point = (df.iloc[df.idxmax()[1]][0], df.iloc[df.idxmax()[1]][1])
    m = (left_point[1] - right_point[1]) / (left_point[0] - right_point[0])
    b = left_point[1] - (m * left_point[0])

    df = df[df['y'] >= (m * df['x'] + b)]
    df.sort_values(by='x', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Cleanup - insure accuracy is always increasing with latency up the Pareto front
    best_acc = 0
    drop_list = []
    for i in range(len(df)):
        if df.iloc[i]['y'] > best_acc:
            best_acc = df.iloc[i]['y']
        else:
            drop_list.append(i)
    df.drop(df.index[drop_list], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.columns = optimization_metrics

    return df


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
        print(cloud[:5])
        alpha_shape = alphashape.alphashape(cloud, 0.0)
        print(alpha_shape)
        ax.add_patch(PolygonPatch(alpha_shape, alpha=0.2))
        # ax.add_patch(
        #     PolygonPatch(
        #         alpha_shape,
        #         fill=None,
        #         alpha=0.4,
        #         color='grey',
        #         linewidth=1.5,
        #         label='Random search boundary',
        #         linestyle='--',
        #     )
        # )

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


def load_csv(
    filepath,
    col_list=['config', 'date', 'params', 'latency', 'macs', 'accuracy_top1'],
    normalize=False,
    idx_slicer=None,
    fit=False,
    scaler=None,
    sort=False,
    verbose=False,
):
    # Sub-network,Date,Model Parameters,Latency (ms),MACs,SST-2 Acc
    if idx_slicer is not None:
        df = pd.read_csv(filepath).iloc[:idx_slicer]
    else:
        df = pd.read_csv(filepath)

    df.columns = col_list

    if sort:
        df = df.sort_values(by=['macs']).reset_index(drop=True)
    if verbose:
        print(filepath)
        print('dataset length: {}'.format(len(df)))
        print('acc max = {}'.format(df['accuracy_top1'].max()))
        print('lat min = {}'.format(df['accuracy_top1'].min()))

    df = df[['macs', 'accuracy_top1']]

    if normalize:
        if fit == True:
            scaler = MinMaxScaler()
            scaler.fit(df['macs'].values.reshape(-1, 1))
            df['macs'] = scaler.transform(df['macs'].values.reshape(-1, 1)).squeeze()
            return df, scaler
        else:
            df['macs'] = scaler.transform(df['macs'].values.reshape(-1, 1)).squeeze()
            return df
    else:
        return df


def collect_hv(hv, supernet):
    start_interval = np.array(list(range(10, 200, 10)))
    end_interval = np.array(list(range(200, 10000, 100)))
    full_interval = np.concatenate([start_interval, end_interval])
    hv_list = list()

    for evals in tqdm.tqdm(full_interval):
        front = frontier_builder(supernet.iloc[:evals], optimization_metrics=['macs', 'accuracy_top1'])
        front['naccuracy_top1'] = -front['accuracy_top1']

        hv_list.append(hv.do(front[['macs', 'naccuracy_top1']].values))

    for i in range(0, len(hv_list) - 1):
        if hv_list[i + 1] < hv_list[i]:
            hv_list[i + 1] = hv_list[i]

    full_interval = np.insert(full_interval, 0, 1, axis=0)
    hv_list = np.array(hv_list)
    hv_list = np.insert(hv_list, 0, 0, axis=0)

    return hv_list, full_interval


def plot_hv():
    plot_subtitle = 'BERT SST-2 {step} ({samples}/{evaluations})'
    save_dir = 'output_2/'
    population = 50
    EVALUATIONS = 2000
    xlim = None
    ylim = None #(73.0, 77.5)
    avg_time = 5.9
    ref_x = [6e9]
    ref_y = [0.90]

    df_linas = load_csv('results/bert_sst2_linas_2000_37.csv', normalize=False, idx_slicer=EVALUATIONS)
    df_nsga = load_csv('results/bert_sst2_nsga2_0.csv', normalize=False, idx_slicer=EVALUATIONS)
    df_random = load_csv('results/bert_sst2_random_0.csv', normalize=False, idx_slicer=EVALUATIONS)
    xlabel = 'MACs'

    df_linas_front = frontier_builder(df_linas, optimization_metrics=['macs', 'accuracy_top1'])
    df_nsga_front = frontier_builder(df_nsga, optimization_metrics=['macs', 'accuracy_top1'])

    evals_list = np.array(list(range(10, 10000, 10)))
    ref_point = [ref_x[0], -ref_y[0]]  # latency, -top1
    hv = HV(ref_point=np.array(ref_point))
    edge_points = []

    ## LINAS
    df_seed1 = load_csv('results/bert_sst2_linas_2000_37.csv', idx_slicer=EVALUATIONS)
    hv_seed1, interval = collect_hv(hv, df_seed1)
    df_seed2 = load_csv('results/bert_sst2_linas_2000_47.csv', idx_slicer=EVALUATIONS)
    hv_seed2, _ = collect_hv(hv, df_seed2)
    df_seed3 = load_csv('results/bert_sst2_linas_2000_57.csv', idx_slicer=EVALUATIONS)
    hv_seed3, _ = collect_hv(hv, df_seed3)

    df_linas_hv = pd.DataFrame(np.vstack((hv_seed1, hv_seed2, hv_seed3)).T)  # Stack all runs from a given search
    df_linas_hv['mean'] = df_linas_hv.mean(axis=1)
    df_linas_hv['std'] = df_linas_hv.std(axis=1)/3**0.5
    edge_points.append(min(df_linas_hv['mean'][population:] - df_linas_hv['std'][population:]))
    edge_points.append(max(df_linas_hv['mean'][population:] + df_linas_hv['std'][population:]))

    ## NSGA2-2
    df_seed1 = load_csv('results/bert_sst2_nsga2_2000_37.csv', idx_slicer=EVALUATIONS)
    hv_seed1, interval = collect_hv(hv, df_seed1)
    df_seed2 = load_csv('results/bert_sst2_nsga2_2000_47.csv', idx_slicer=EVALUATIONS)
    hv_seed2, _ = collect_hv(hv, df_seed2)
    df_seed3 = load_csv('results/bert_sst2_nsga2_2000_57.csv', idx_slicer=EVALUATIONS)
    hv_seed3, _ = collect_hv(hv, df_seed3)

    df_full_hv = pd.DataFrame(np.vstack((hv_seed1, hv_seed2, hv_seed3)).T)
    df_full_hv['mean'] = df_full_hv.mean(axis=1)
    df_full_hv['std'] = df_full_hv.std(axis=1)/3**0.5
    edge_points.append(min(df_full_hv['mean'][population:] - df_full_hv['std'][population:]))
    edge_points.append(max(df_full_hv['mean'][population:] + df_full_hv['std'][population:]))

    ## RANDOM
    df_seed1 = load_csv('results/bert_sst2_random_2000_37.csv', idx_slicer=EVALUATIONS)
    hv_seed1, interval = collect_hv(hv, df_seed1)
    df_seed2 = load_csv('results/bert_sst2_random_2000_47.csv', idx_slicer=EVALUATIONS)
    hv_seed2, _ = collect_hv(hv, df_seed2)
    df_seed3 = load_csv('results/bert_sst2_random_2000_57.csv', idx_slicer=EVALUATIONS)
    hv_seed3, _ = collect_hv(hv, df_seed3)

    df_rand_hv = pd.DataFrame(np.vstack((hv_seed1, hv_seed2, hv_seed3)).T)
    df_rand_hv['mean'] = df_rand_hv.mean(axis=1)
    df_rand_hv['std'] = df_rand_hv.std(axis=1)/3**0.5
    edge_points.append(min(df_rand_hv['mean'][population:] - df_rand_hv['std'][population:]))
    edge_points.append(max(df_rand_hv['mean'][population:] + df_rand_hv['std'][population:]))

    ylim_hv = (min(edge_points)-0.05*min(edge_points), max(edge_points)+0.05*min(edge_points))

    os.makedirs(save_dir, exist_ok=True)

    for samples in tqdm.tqdm(range(0, EVALUATIONS+1, population*4)):
        elapsed_total_m = avg_time*samples
        elapsed_h = int(elapsed_total_m//60)
        elapsed_m = int(elapsed_total_m-(elapsed_h*60))
        if PLOT_HYPERVOLUME:
            fig, ax = plt.subplots(1, 3, figsize=(15,5), gridspec_kw={'width_ratios': [2.5, 3, 2.5]})
        else:
            fig, ax = plt.subplots(1, 2, figsize=(10,5), gridspec_kw={'width_ratios': [2.5, 3.0]})
        fig.suptitle(plot_subtitle.format(step=samples//population, samples=samples, evaluations=EVALUATIONS, elapsed_h=elapsed_h, elapsed_m=elapsed_m),
            fontweight ="bold")
        cm = plt.cm.get_cmap('viridis_r')

        # LINAS plot
        df_conc = df_linas
        data=df_conc[:samples][['macs', 'accuracy_top1']]
        count = [x for x in range(len(data))]
        x = data['macs']
        y = data['accuracy_top1']

        ax[0].set_title('DyNAS-T')
        ax[0].scatter(x, y, marker='D', alpha=0.8, c=count, cmap=cm, label='Unique DNN\nArchitecture', s=6)
        ax[0].set_ylabel('Accuracy', fontsize=13)
        ax[0].plot(df_linas_front['macs'], df_linas_front['accuracy_top1'],
                color='red', linestyle='--', label='DyNAS-T Pareto front')
        # ax[0].scatter(ref_x, ref_y, marker='s', color='#c00', label='Reference ResNset50 OV INT8')

        # NSGA-II plot
        data1=df_nsga[:samples][['macs', 'accuracy_top1']]
        print(len(data1))
        count = [x for x in range(len(data1))]
        x = data1['macs']
        y = data1['accuracy_top1']

        ax[1].set_title('NSGA-II')
        ax[1].scatter(x, y, marker='D', alpha=0.8, c=count, cmap=cm, label='Unique DNN Architecture', s=6)

        # ax[1].get_yaxis().set_ticklabels([])
        ax[1].plot(df_nsga_front['macs'], df_nsga_front['accuracy_top1'],
                color='red', linestyle='--', label='DyNAS-T Pareto front')
        # ax[1].scatter(ref_x, ref_y, marker='s', color='#c00', label='Reference ResNset50 OV INT8')


        cloud = list(df_random[['macs','accuracy_top1']].to_records(index=False))
        # alpha_shape = alphashape.alphashape(cloud, 0)


        for ax in fig.get_axes()[:2]:
            # ax.add_patch(PolygonPatch(alpha_shape, fill=None, alpha=0.8, linewidth=1.5, label='Random search boundary', linestyle='--'))


            ax.legend(fancybox=True, fontsize=10, framealpha=1, borderpad=0.2, loc='lower right')
            # if ylim:
            #     ax.set_ylim(ylim)
            # if xlim:
            #     ax.set_xlim(xlim)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel(xlabel, fontsize=13)

        if PLOT_HYPERVOLUME and samples >= population:
            fig.get_axes()[2].set_title('Hypervolume')

            ########## LINAS
            fig.get_axes()[2].plot(interval, df_linas_hv['mean'], label='LINAS', color=colors['red'], linewidth=2)
            fig.get_axes()[2].fill_between(interval, df_linas_hv['mean']-df_linas_hv['std'], df_linas_hv['mean']+df_linas_hv['std'],
                            color=colors['red'], alpha=0.2)
            ########## NSGA / FULL
            fig.get_axes()[2].plot(interval, df_full_hv['mean'], label='NSGA-II', linestyle='--', color=colors['blue'], linewidth=2)
            fig.get_axes()[2].fill_between(interval, df_full_hv['mean']-df_full_hv['std'], df_full_hv['mean']+df_full_hv['std'],
                            color=colors['blue'], alpha=0.2)
            ########## RANDOM
            fig.get_axes()[2].plot(interval, df_rand_hv['mean'], label='Random Search', linestyle='-.', color=colors['orange'], linewidth=2)
            fig.get_axes()[2].fill_between(interval, df_rand_hv['mean']-df_rand_hv['std'], df_rand_hv['mean']+df_rand_hv['std'],
                            color=colors['orange'], alpha=0.2)
            ##########

            fig.get_axes()[2].set_xlim(population, samples)
            # if ylim_hv:
            #     fig.get_axes()[2].set_ylim(ylim_hv)

            fig.get_axes()[2].set_xlabel('Evaluation Count', fontsize=13)
            fig.get_axes()[2].set_ylabel('Hypervolume', fontsize=13)
            fig.get_axes()[2].legend(fancybox=True, fontsize=12, framealpha=1, borderpad=0.2, loc='best')

            fig.get_axes()[2].grid(True, alpha=0.2)


            formatter = ScalarFormatter()
            formatter.set_scientific(False)
            fig.get_axes()[2].xaxis.set_major_formatter(formatter)

        # Eval Count bar
        norm = plt.Normalize(0, len(data))
        sm = ScalarMappable(norm=norm, cmap=cm)
        cbar = fig.colorbar(sm, ax=ax, shrink=0.85)
        cbar.ax.set_title("         Evaluation\n  Count", fontsize=8)

        fig.tight_layout(pad=1)
        plt.subplots_adjust(wspace=0.07, hspace=0)
        plt.show();
        fn = save_dir + '/pareto_{}.png'.format(samples//population)

        fig.savefig(fn, bbox_inches='tight', pad_inches=0, dpi=150)

# def correlation() -> None:
#     df_1 = pd.read_csv('bnas_1_random.csv')[:50]
#     df_2 = pd.read_csv('bnas_2_random.csv')[:50]
#     df_1.columns = ['config', 'date', 'params', 'latency', 'macs', 'accuracy_top1']
#     df_2.columns = ['config', 'date', 'params', 'latency', 'macs', 'accuracy_top1']

#     plot_correlation(x1=df_1['accuracy_top1'], x2=df_2['accuracy_top1'])


if __name__ == '__main__':
    # plot_search_progression(
    #     # results_path='bnas_1_latency.csv',
    #     # results_path='results_tlt_linas_dist2_long.csv',
    #     # results_path='results_ofambv3_random_long.csv',
    #     results_path='bootstrapnas_resnet50_cifar10_linas.csv',
    #     # random_results_path='test.csv',
    #     # random_results_path='bnas_2_random.csv',
    #     # random_results_path='bnas_mbv2_cifar_top1_macs_random.csv',
    # )

    # plot_search_progression(
    #     results_path='bootstrapnas_resnet50_cifar10_random.csv',
    # )
    #  plot_search_progression(results_path='test_nsga.csv')
    #  plot_search_progression(results_path='test_nsga2.csv')
    #  plot_search_progression(results_path='test_random.csv')
    #  plot_search_progression(results_path='test.csv')
    if False:
        plot_search_progression(results_path='bert_sst2_linas.csv')  # , random_results_path='bert_sst2_random.csv')
        plot_search_progression(results_path='bert_sst2_nsga2.csv')  # , random_results_path='bert_sst2_random.csv')
        plot_search_progression(results_path='bert_sst2_random.csv')
    plot_hv()


# correlation()
