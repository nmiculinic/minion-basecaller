import importlib
import os
from sigopt import Connection
from dotenv import load_dotenv, find_dotenv
from time import monotonic
import model_utils
import argparse
import json

load_dotenv(find_dotenv())

recovery_file = os.path.join(model_utils.repo_root, 'log', 'recovery_%s.json' % model_utils.hostname)
print(recovery_file)


def sigopt_runner(module_name=None, observation_budget=20, train_steps=100000):
    parser = argparse.ArgumentParser()
    if module_name is None:
        parser.add_argument("module_name", type='str', help='Model name')
    parser.add_argument("train_steps", nargs='?', type=int,
                        default=train_steps, help='Number of training steps')
    parser.add_argument('--budget', type=int,
                        default=observation_budget)
    parser.add_argument('--batch_size', '-b', type=int,
                        default=-1, help="batch_size")
    parser.add_argument('--num_workers', type=int,
                        default=3, help='Number of worker threads for feeding queues')
    parser.add_argument("-s", "--summarize", help="Summarize gradient during training", action="store_true")
    parser.add_argument('--name', type=str,
                        default=module_name, help="Model name [run_id]", dest="model_name")
    args = parser.parse_args()

    model_module = importlib.import_module(module_name)
    print("Importing %s" % module_name)
    content = dir(model_module)
    conn = Connection(client_token=os.environ["SIGOPT_KEY"])
    if os.environ["EXPERIMENT_ID"] == "NEW":
        experiment = conn.experiments().create(
            name='MinION basecaller residual',
            parameters=model_module.params,
            observation_budget=args.budget
        )
        print("Created experiment: https://sigopt.com/experiment/" +
              experiment.id, "Budget %d" % args.budget)
        experiment_id = experiment.id
    else:
        experiment_id = os.environ["EXPERIMENT_ID"]
        print("Using experiment: https://sigopt.com/experiment/" + experiment_id)

    if "verify_hyper" in content:
        print("Using module verify_hyper")
        verify_hyper = model_module.verify_hyper
    else:
        print("Using default verify hyperparameters")
        verify_hyper = lambda x: True

    while True:
        if os.path.exists(recovery_file):
            with open(recovery_file, 'r') as f:
                print("Reloading EXISTING model params!!!")
                params = json.load(f)
                hyper = params['hyper']
                suggestion_id = params['suggestion_id']
                reuse = True
        else:
            # Choose hyperparameters
            suggestion = conn.experiments(experiment_id).suggestions().create()
            hyper = dict(suggestion.assignments)

            while not verify_hyper(hyper):
                print("Rejecting suggestion:")
                for k in sorted(hyper.keys()):
                    print("%-20s: %7s" % (k, str(hyper[k])))
                conn.experiments(experiment_id).observations().create(
                    suggestion=suggestion.id,
                    metadata=dict(
                        hostname=model_utils.hostname,
                    ),
                    failed=True
                )
                suggestion = conn.experiments(experiment_id).suggestions().create()
                hyper = dict(suggestion.assignments)
            suggestion_id = suggestion.id
            reuse = False

            if os.environ['SIGOPT_KEY'].startswith("TJEAVRLBP"):
                print("DEVELOPMENT MODE!!!")
                hyper.update(model_module.default_params)
            else:
                print("PRODUCTION MODE!!!")

        print("Running hyper parameters")
        for k in sorted(hyper.keys()):
            print("%-20s: %7s" % (k, str(hyper[k])))

        # Setup model
        model_params = model_module.model_setup_params(hyper)
        model_params['run_id'] = args.model_name + \
            "_%s_%s" % (experiment_id, suggestion_id)

        if args.batch_size != -1:
            model_params['batch_size'] = args.batch_size
        model_params['reuse'] = reuse

        with open(recovery_file, 'w') as f:
            json.dump({
                'hyper': hyper,
                'suggestion_id': suggestion_id,
            }, f, sort_keys=True, indent=4)
        start_timestamp = monotonic()
        model = model_utils.Model(**model_params)
        if reuse:
            model.logger.info("Reusing model for earlier crash")
        result = model.simple_managed_train_model(
            args.train_steps, summarize=args.summarize, num_workers=args.num_workers)

        avg_acc = result['accuracy']['mu']
        se = result['accuracy']['se']
        print("reporting to sigopt:", avg_acc, se, type(avg_acc), type(se))
        # Final reporting
        conn.experiments(experiment_id).observations().create(
            suggestion=suggestion_id,
            value=avg_acc,
            metadata=dict(
                hostname=model_utils.hostname,
                suggestion_id=suggestion_id,
                result=str(result),
                **{
                    'time[h]': (monotonic() - start_timestamp) / 3600.0,
                    'logdir': model.log_dir,
                },
            ),
            value_stddev=se
        )
        os.remove(recovery_file)


if __name__ == "__main__":
    sigopt_runner()
