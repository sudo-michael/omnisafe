from omnisafe.common.statistics_tools import StatisticsTools


# just fill in the path in which experiment grid runs.
PATH = './exp-x/omnisafe_test_benchmark_ppo'
if __name__ == '__main__':
    st = StatisticsTools()
    st.load_source(PATH)
    # just fill in the name of the parameter of which value you want to compare.
    # then you can specify the value of the parameter you want to compare,
    # or you can just specify how many values you want to compare in single graph at most,
    # and the function will automatically generate all possible combinations of the graph.
    # but the two mode can not be used at the same time.
    # `['SafetyAntVelocity-v1', 'SafetyHopperVelocity-v1', 'SafetyHalfCheetahVelocity-v1']
    st.draw_graph(parameter='algo', values=None, compare_num=1, cost_limit=25, show_image=False)
    # st.draw_graph(parameter='env_id', values=['P3O'], compare_num=None, cost_limit=25, show_image=False)
# import argparse

# from omnisafe.utils.plotter import Plotter


# # For example, you can run the following command to plot the training curve:
# # python plot.py --logdir omnisafe/examples/runs/PPOLag-{SafetyAntVelocity-v1}
# # after training the policy with the following command:
# # python train_policy.py --algo PPOLag --env-id SafetyAntVelocity-v1
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--logdir', nargs='*')
#     parser.add_argument('--legend', '-l', nargs='*')
#     parser.add_argument('--xaxis', '-x', default='Steps')
#     parser.add_argument('--value', '-y', default='Rewards', nargs='*')
#     parser.add_argument('--count', action='store_true')
#     parser.add_argument('--smooth', '-s', type=int, default=1)
#     parser.add_argument('--select', nargs='*')
#     parser.add_argument('--exclude', nargs='*')
#     parser.add_argument('--estimator', default='mean')
#     args = parser.parse_args()

#     plotter = Plotter()
#     plotter.make_plots(
#         args.logdir,
#         args.legend,
#         args.xaxis,
#         args.value,
#         args.count,
#         smooth=args.smooth,
#         select=args.select,
#         exclude=args.exclude,
#         estimator=args.estimator,
#     )