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

import alphashape
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from descartes import PolygonPatch
from matplotlib.cm import ScalarMappable
from scipy.spatial import Delaunay
from shapely.geometry import MultiLineString, MultiPoint, mapping
from shapely.ops import cascaded_union, polygonize

from dynast.utils import log


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
    plot_search_progression(results_path='bert_sst2_linas.csv')#, random_results_path='bert_sst2_random.csv')
    plot_search_progression(results_path='bert_sst2_nsga2.csv')#, random_results_path='bert_sst2_random.csv')
    plot_search_progression(results_path='bert_sst2_random.csv')

# correlation()
