# Copyright 2023 OmniSafe Team. All Rights Reserved.
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
"""Implementation of the statistics tools."""

from __future__ import annotations

import itertools
import json
import os
from copy import deepcopy
from typing import Any, Generator

from omnisafe.utils.plotter import Plotter
from omnisafe.utils.tools import assert_with_exit, hash_string, recursive_dict2json, update_dict


class StatisticsTools:
    """Analyze experiments results launched by experiment grid.

    Users can choose any parameters to compare the results. Aiming to help users to find the best
    hyperparameter faster.

    Attributes:
        grid_config_dir (str): The directory of grid_config.json.
        decompressed_grid_config (dict[str, Any]): The decompressed grid_config.json.
        path_map_img_name (dict[str, Any]): The map from path to image name.
        grid_config (dict[str, Any]): The grid_config.json.
        exp_dir (str): The experiment directory.
        plotter (Plotter): The plotter.
    """

    grid_config_dir: str
    decompressed_grid_config: dict[str, Any]
    path_map_img_name: dict[str, Any]
    grid_config: dict[str, Any]
    exp_dir: str

    def __init__(self) -> None:
        """Initialize an instance of :class:`StatisticsTools`."""
        self.plotter: Plotter = Plotter()

    def load_source(self, path: str) -> None:
        """Load experiment results.

        Args:
            path (str): The experiment directory.
        """
        # recursively find directory which is generated by experiment grid
        grid_config_dirs = []
        for root, _, files in os.walk(path):
            if 'grid_config.json' in files:
                grid_config_dirs.append(files)
                self.grid_config_dir = os.path.join(root, 'grid_config.json')
                self.exp_dir = root

        assert_with_exit(
            hasattr(self, 'grid_config_dir'),
            'cannot find directory which is initialized by experiment grid via grid_config.json',
        )
        assert_with_exit(
            len(grid_config_dirs) == 1,
            'there should be only one experiment grid directory',
        )

        # load the config file of experiment grid
        try:
            with open(self.grid_config_dir, encoding='utf-8') as file:
                self.grid_config = json.load(file)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                'The config file is not found in the save directory.',
            ) from error

    def draw_graph(
        self,
        parameter: str,
        values: list[Any] | None = None,
        compare_num: int | None = None,
        cost_limit: float | None = None,
        smooth: int = 1,
        show_image: bool = False,
    ) -> None:
        """Draw graph.

        Args:
            parameter (str): The parameter to compare.
            values (list[Any] or None, optional): The values of the parameter to compare. Defaults
                to None.
            compare_num (int or None, optional): The number of values to compare. Defaults to None.
            cost_limit (float or None, optional): The cost limit of the experiment. Defaults to None.
            smooth (int, optional): The smooth window size. Defaults to 1.
            show_image (bool): Whether to show graph image in GUI windows.

        .. note::
            `values` and `compare_num` cannot be set at the same time.
        """
        # check whether operation is valid
        assert_with_exit(
            not (values and compare_num),
            'values and compare_num cannot be set at the same time',
        )
        assert_with_exit(hasattr(self, 'grid_config'), 'please load source first')
        assert_with_exit(
            parameter in self.grid_config,
            f'parameter scope `{parameter}` is not in {self.grid_config}',
        )

        # decompress the grid config
        decompressed_cfgs: dict = {}
        for k, v in self.grid_config.items():
            update_dict(decompressed_cfgs, self.decompress_key(k, v))
        self.decompressed_grid_config = decompressed_cfgs
        parameter_values = self.get_compressed_key(self.decompressed_grid_config, parameter)

        # make config groups via the combination of parameter values
        if not (values or compare_num):
            compare_num = len(parameter_values)
        graph_paths = self.make_config_groups(parameter, parameter_values, values, compare_num)

        for graph_dict in graph_paths:
            legend = []
            log_dirs = []
            img_name_cfgs = {}
            for (_, value), path in graph_dict.items():
                legend += [f'{value}']
                log_dirs += [path]
            img_name_cfgs = self.path_map_img_name[list(graph_dict.values())[-1]]
            decompressed_img_name_cfgs: dict = {}
            for k, v in img_name_cfgs.items():
                update_dict(decompressed_img_name_cfgs, self.decompress_key(k, v[0]))
            save_name = (
                list(graph_dict.keys())[-1][0][:10]  # pylint: disable=undefined-loop-variable
                + '---'
                + decompressed_img_name_cfgs['env_id'][:30]
                + '---'
                + hash_string(recursive_dict2json(decompressed_img_name_cfgs))
            )
            # breakpoint()
            try:
                self.plotter.make_plots(
                    log_dirs,
                    legend,
                    'Steps',
                    'Rewards',
                    False,
                    cost_limit,
                    smooth,
                    None,
                    None,
                    'mean',
                    save_name=save_name,
                    show_image=show_image,
                )
            except Exception:  # noqa # pragma: no cover # pylint: disable=broad-except
                print(
                    f'Cannot generate graph for {save_name[:5] + str(decompressed_img_name_cfgs)}',
                )
                print(Exception)

    def make_config_groups(
        self,
        parameter: str,
        parameter_values: list[str],
        values: list[Any] | None = None,
        compare_num: int | None = None,
    ) -> list[dict[tuple[str, Any], str]]:
        """Make config groups.

        Each group contains a list of config paths to compare.

        .. warning::
            `values` and `compare_num` cannot be set at the same time.

        Args:
            parameter (str): The parameter to compare.
            parameter_values (list[str]): The values of the parameter to compare.
            values (list[Any] or None, optional): The values of the parameter to compare. Defaults
                to None.
            compare_num (int or None, optional): The number of values to compare. Defaults to None.

        Returns:
            A list of graph paths.
        """
        self.path_map_img_name = {}
        parameter_values_combination: list[tuple] = []
        graph_groups: list[list] = []
        assert (values is not None) ^ (
            compare_num is not None
        ), 'The values and compare_num cannot be set at the same time'
        if values:
            assert_with_exit(
                all(v in parameter_values for v in values),
                f'values `{values}` of parameter `{parameter}` is not subset of `{parameter_values}`',
            )
            # if values is specified, will only compare values in it
            parameter_values_combination = [tuple(values)]
        if compare_num:
            assert_with_exit(
                compare_num <= len(parameter_values),
                (
                    f'compare_num `{compare_num}` is larger than number of values '
                    f'`{len(parameter_values)}` of parameter `{parameter}`'
                ),
            )
            # if compare_num is specified, will combine any potential combination to compare
            parameter_values_combination = list(self.combine(parameter_values, compare_num))
        group_config = deepcopy(self.grid_config)
        # value of parameter is determined above
        group_config.pop(parameter)
        # seed is not a parameter
        if 'seed' in group_config:
            group_config.pop('seed')
        if 'train_cfgs' in group_config:
            group_config['train_cfgs'].pop('device', None)
        # combine all possible combinations of other parameters
        # fix them in a single graph and only vary values of parameter which is specified by us
        for pinned_config in self.dict_permutations(group_config):
            group_config.update(pinned_config)
            for compare_value in parameter_values_combination:
                group_config[parameter] = list(compare_value)
                img_name_cfgs = deepcopy(group_config)
                graph_groups.append(
                    [
                        img_name_cfgs,
                        self.variants(list(group_config.keys()), list(group_config.values())),
                    ],
                )

        graph_paths = []
        for img_name_cfgs, graph in graph_groups:
            paths = {}
            for path_dict in graph:
                exp_name = (
                    path_dict['env_id'][:30] + '---' + hash_string(recursive_dict2json(path_dict))
                )
                path = os.path.join(self.exp_dir, exp_name)
                self.path_map_img_name[path] = img_name_cfgs
                para_val = (parameter, self.get_compressed_key(path_dict, parameter))
                paths[para_val] = path
            graph_paths.append(paths)

        return graph_paths

    def decompress_key(self, compressed_key: str, value: Any) -> dict[str, Any]:
        """This function is used to convert the custom configurations to dict.

        .. note::
            This function is used to convert the custom configurations to dict. For example, if the
            custom configurations are ``train_cfgs:use_wandb`` and ``True``, then the output dict
            will be ``{'train_cfgs': {'use_wandb': True}}``.

        Args:
            compressed_key (str): The compressed key.
            value (Any): The value of the compressed key.

        Returns:
            The decompressed dict.
        """
        keys_split = compressed_key.replace('-', '_').split(':')
        return_dict = {keys_split[-1]: value}

        for key in reversed(keys_split[:-1]):
            return_dict = {key.replace('-', '_'): return_dict}
        return return_dict

    def _variants(self, keys: list[str], vals: list[Any]) -> list[dict[str, Any]]:
        """Recursively builds list of valid variants.

        Args:
            keys (list[str]): The keys of the config.
            vals (list[Any]): The values of the config.

        Returns:
            List of valid variants.
        """
        if len(keys) == 1:
            pre_variants: list[dict[str, Any]] = [{}]
        else:
            pre_variants = self._variants(keys[1:], vals[1:])

        variants = []
        for val in vals[0]:
            for pre_v in pre_variants:
                current_variants = deepcopy(pre_v)
                v_temp = {}
                key_list = keys[0].split(':')
                v_temp[key_list[-1]] = val
                for key in reversed(key_list[:-1]):
                    v_temp = {key: v_temp}
                self.update_dict(current_variants, v_temp)
                variants.append(current_variants)

        return variants

    def update_dict(self, total_dict: dict[str, Any], item_dict: dict[str, Any]) -> None:
        """Updater of multi-level dictionary.

        Args:
            total_dict (dict[str, Any]): The total dictionary.
            item_dict (dict[str, Any]): The item dictionary.
        """
        for idd in item_dict:
            total_value = total_dict.get(idd)
            item_value = item_dict.get(idd)

            if total_value is None:
                total_dict.update({idd: item_value})
            elif isinstance(item_value, dict):
                self.update_dict(total_value, item_value)
                total_dict.update({idd: total_value})
            else:
                total_value = item_value
                total_dict.update({idd: total_value})

    def variants(self, keys: list[str], vals: list[Any]) -> list[dict[str, Any]]:
        """Makes a list of dict, where each dict is a valid config in the grid.

        There is special handling for variant parameters whose names take the form

            ``'full:param:name'``

        The colons are taken to indicate that these parameters should have a nested dict structure.
        For example, if there are two params,

        ====================  ===
        Key                   Val
        ====================  ===
        ``'base:param:a'``    1
        ``'base:param:b'``    2
        ====================  ===

        the variant dict will have the structure

        .. parsed-literal::

            variant = {
                base: {
                    param : {
                        a : 1,
                        b : 2
                        }
                    }
                }

        Args:
            keys (list[str]): The keys of the config.
            vals (list[Any]): The values of the config.

        Returns:
            List of valid and not duplicate variants.
        """
        flat_variants = self._variants(keys, vals)

        def check_duplicate(var: dict[str, Any]) -> dict[str, Any]:
            """Build the full nested dict version of var, based on key names."""
            new_var: dict = {}

            for key, value in var.items():
                assert key not in new_var, "You can't assign multiple values to the same key."
                new_var[key] = value

            return new_var

        return [check_duplicate(var) for var in flat_variants]

    def combine(self, sequence: list[str], num_choosen: int) -> Generator:
        """Combine elements in sequence to n elements.

        Args:
            sequence (list[str]): The sequence to be combined.
            num_choosen (int): The number of elements to be combined.

        Returns:
            The generator of the combined elements.
        """
        if num_choosen == 1:
            for i in sequence:
                yield (i,)
        else:
            for idx, item in enumerate(sequence):
                for nxt in self.combine(sequence[idx + 1 :], num_choosen - 1):
                    yield (item, *nxt)

    def dict_permutations(self, input_dict: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate all possible combinations of the values in a dictionary.

        Takes a dictionary with string keys and list values, and returns a dictionary with all
        possible combinations of the lists as values for each key.

        Args:
            input_dict (dict[str, Any]): The input dictionary.

        Returns:
            The list of all possible combinations of the values in a dictionary.
        """
        keys = list(input_dict.keys())
        values = list(input_dict.values())
        value_combinations = list(itertools.product(*values))

        result = []
        for combination in value_combinations:
            new_dict = {}
            for i, item in enumerate(keys):
                new_dict[item] = [combination[i]]
            result.append(new_dict)

        return result

    def get_compressed_key(self, dictionary: dict[str, Any], key: str) -> Any:
        """Get the compressed value of the key.

        Args:
            dictionary (dict[str, Any]): the uncompressed dictionary.
            key (str): the key.

        Returns:
            The compressed value of the key.
        """
        inner_config = dictionary
        for k in key.split(':'):
            inner_config = inner_config[k]
        return inner_config
