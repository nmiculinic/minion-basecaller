import os
from dotenv import load_dotenv, find_dotenv
import model_utils
import argparse
import sys
from sigopt import Connection
from time import monotonic
import json
import input_readers
import util
from time import perf_counter
load_dotenv(find_dotenv())

usage = """
    This File is responsible for controlling model training, testing and basecalling.

    python <model_file> (train|basecall|eval) (task specific arguments, -h for help)
"""


def control(context):
    if len(sys.argv) <= 1:
        print("too few arguments!")
        print(usage)
        sys.exit(1)
    task = sys.argv[1]
    del sys.argv[1]
    if task == "train":
        sigopt_runner(**context)
    elif task == "basecall":
        basecall(**context)
    elif task == "eval":
        eval_model(**context)
    else:
        print("Unknown task type, {}".format(task))
        print(usage)
        sys.exit(2)


def eval_model(create_test_model, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="increase output verbosity", type=str)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("-c", "--checkpoint", help="Checkpoint to restore", type=str, default=None)
    parser.add_argument("count", nargs='?', type=int, default=-1, help='Number of evaluation count from test set. Default -1 meaning whole test set')
    parser.add_argument("--fasta_out", "-o", type=str, default=None, help='Directory for output fasta files from processed fast5 files')
    parser.add_argument("--ref", type=str, default=None, help='Path to reference string')

    args = parser.parse_args()

    with open(os.path.join(args.model_dir, 'model_hyperparams.json'), 'r') as f:
        hyper = json.load(f)

    model = create_test_model(log_dir=args.model_dir, reuse=True, overwrite=False, hyper=hyper)

    try:
        model.init_session(start_queues=False)
        model.restore(checkpoint=args.checkpoint)
        count = args.count
        if count == -1:
            count = 1.0

        if args.fasta_out is None and args.ref is not None:
            args.fasta_out = os.path.join(model.log_dir, 'fasta')
            os.makedirs(args.fasta_out, exist_ok=True)
        model.run_validation_full(frac=count, verbose=args.verbose, fasta_out_dir=args.fasta_out, ref=args.ref)
    finally:
        model.close_session()


def basecall(create_test_model, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="increase output verbosity", type=str)
    parser.add_argument("fast5_in", help="Fast5 file to basecall or dir of fast5 files", default='.', type=str)
    parser.add_argument("out_dir", nargs='?', type=str, default=None, help='Directory for output fasta files from processed fast5 files')
    parser.add_argument("-c", "--checkpoint", help="Checkpoint to restore", type=str, default=None)

    args = parser.parse_args()
    if os.path.isfile(args.fast5_in):
        file_list = [args.fast5_in]
    elif os.path.isdir(args.fast5_in):
        file_list = [os.path.join(args.fast5_in, path) for path in os.listdir(args.fast5_in) if os.path.splitext(path)[1] == ".fast5"]
    else:
        print("Not file not dir %s, exiting!!!" % args.fast5_in)
        sys.exit(1)

    with open(os.path.join(args.model_dir, 'model_hyperparams.json'), 'r') as f:
        hyper = json.load(f)

    model = create_test_model(log_dir=args.model_dir, reuse=True, overwrite=False, hyper=hyper)

    try:
        model.init_session(start_queues=False)
        model.restore(checkpoint=args.checkpoint)
        if args.out_dir is not None:
            os.makedirs(args.out_dir, exist_ok=True)
        total_time = 0
        total_bases = 0
        for f in file_list:

            if args.out_dir is not None:
                out = os.path.splitext(f)[0].split('/')[-1] + ".fasta"
                out = os.path.join(args.out_dir, out)
            else:
                out = None
            t0 = perf_counter()
            basecalled = model.basecall_sample(f, fasta_out=out)
            total_time += perf_counter() - t0
            total_bases += len(basecalled)
            print("Speed %.3f bps" % (total_bases / total_time), file=sys.stderr)

            if out is None:
                util.dump_fasta(os.path.splitext(f)[0].split(os.sep)[-1], basecalled, sys.stdout)

    finally:
        model.close_session()


def sigopt_runner(
    create_train_model,
    sigopt_params,
    default_params,
    default_name,
    verify_hyper=lambda x: True,
    **other
):
    recovery_file = os.path.join(
        model_utils.repo_root, 'log', 'recovery_%s.json' % model_utils.hostname)
    print(recovery_file)

    parser = argparse.ArgumentParser()
    parser.add_argument("train_steps", nargs='?', type=int,
                        default=50000, help='Number of training steps')
    parser.add_argument('--budget', type=int,
                        default=20)
    parser.add_argument('--batch_size', '-b', type=int,
                        default=-1, help="batch_size")
    parser.add_argument('--num_workers', type=int,
                        default=3, help='Number of worker threads for feeding queues')
    parser.add_argument(
        "-s", "--summarize", help="Summarize gradient during training", action="store_true")
    parser.add_argument('--name', type=str,
                        default=default_name, help="Model name [run_id]", dest="model_name")
    parser.add_argument('--trace_every', '-t', type=int,
                        default=10000, help="Each x steps to run profile trace. Negative number (e.g. -1) to disable")
    parser.add_argument("--ref", type=str, default=None,
                        help='Path to reference string')
    args = parser.parse_args()

    if args.ref is None:
        suggested_ref = os.path.join(
            input_readers.root_dir_default, 'reference2.fasta')

        if os.path.isfile(suggested_ref):
            args.ref = suggested_ref
            print("Using %s for reference" % args.ref)
        else:
            print(
                "Cannot find default reference at %s, ignoring Graphmap etc." % suggested_ref)

    conn = Connection(client_token=os.environ["SIGOPT_KEY"])
    if os.environ["EXPERIMENT_ID"] == "NEW":
        experiment = conn.experiments().create(
            name='MinION basecaller residual',
            parameters=sigopt_params,
            observation_budget=args.budget
        )
        print("Created experiment: https://sigopt.com/experiment/" +
              experiment.id, "Budget %d" % args.budget)
        experiment_id = experiment.id
    else:
        experiment_id = os.environ["EXPERIMENT_ID"]
        print("Using experiment: https://sigopt.com/experiment/" + experiment_id)

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
                suggestion = conn.experiments(
                    experiment_id).suggestions().create()
                hyper = dict(suggestion.assignments)
            suggestion_id = suggestion.id
            reuse = False

            if os.environ['SIGOPT_KEY'].startswith("TJEAVRLBP"):
                print("DEVELOPMENT MODE!!!")
                hyper.update(default_params)
            else:
                print("PRODUCTION MODE!!!")

        print("Running hyper parameters")
        for k in sorted(hyper.keys()):
            print("%-20s: %7s" % (k, str(hyper[k])))

        # Setup model
        model_extra_params = {}
        model_extra_params['run_id'] = args.model_name + \
            "_%s_%s" % (experiment_id, suggestion_id)

        if args.batch_size != -1:
            model_extra_params['batch_size'] = args.batch_size
        model_extra_params['reuse'] = reuse

        with open(recovery_file, 'w') as f:
            json.dump({
                'hyper': hyper,
                'suggestion_id': suggestion_id,
            }, f, sort_keys=True, indent=4)
        start_timestamp = monotonic()
        model = create_train_model(hyper, **model_extra_params)
        if reuse:
            model.logger.info("Reusing model for earlier crash")
        result = model.simple_managed_train_model(
            args.train_steps, summarize=args.summarize, num_workers=args.num_workers, trace_every=args.trace_every, ref=args.ref)

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
