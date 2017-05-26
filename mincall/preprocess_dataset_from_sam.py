import glob
import logging
import os
from argparse import ArgumentParser
from configparser import ConfigParser
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
import h5py
from mincall import align_utils, bioinf_utils
from mincall.preprocess_dataset import get_ref_starts_dict, _dump_ref_files, reads_train_test_split, \
    _make_rel_symlink, _make_train_test_symlinks
load_dotenv(find_dotenv())


def construct_ref_files(fast5_in, sam_path, out_dir, ref_path):
    if os.path.isfile(fast5_in):
        file_list = [fast5_in]
    elif os.path.isdir(fast5_in):
        file_list = glob.glob(os.path.join(fast5_in, '*.fast5'))
    else:
        logging.error("Invalid fastin - expected file or dir %s, skipping!!!" % fast5_in)
        return

    os.makedirs(out_dir, exist_ok=True)
    ref_starts = get_ref_starts_dict(ref_path)
    _construct_ref_files(file_list, sam_path, ref_starts, out_dir)


def _construct_ref_files(fast5_files, sam_path, ref_starts, out_root):
    if not fast5_files:
        return
    name_to_file = {}
    total_skipped = 0

    total_files = len(fast5_files)
    pbar = tqdm(fast5_files, desc='Loading reads from fast5')
    for f in pbar:
        try:
            with h5py.File(f, 'r') as h5:
                template_key = '/Analyses/Basecall_1D_000/BaseCalled_template/Fastq'
                if template_key not in h5:
                    total_skipped += 1
                    continue
                fastq = h5[template_key][()]
                read_name, *_ = fastq.strip().split(b'\n')
                read_name = read_name[1:].split(b' ')[0].decode()
                assert read_name not in name_to_file
                name_to_file[read_name] = os.path.basename(f)
        except Exception as ex:
            logging.error('Error reading file %s', f, exc_info=True)

        pbar.set_postfix(stat="Total skipped with no fastq: %d/%d" % (total_skipped, total_files))

    logging.debug("Total skipped with no fastq: %d/%d", total_skipped, total_files)

    logging.debug("Finding references in sam")
    result_dict = align_utils.get_target_sequences(sam_path)

    logging.debug("Dumping results")
    _dump_ref_files(result_dict, name_to_file, ref_starts, out_root)


def preprocess_all_ref(dataset_conf_path, only_split):
    dataset_config = ConfigParser()
    dataset_config.read(dataset_conf_path)
    train_root = dataset_config['DEFAULT']['train_root']
    test_root = dataset_config['DEFAULT']['test_root']
    per_genome_split_root = dataset_config['DEFAULT']['per_genome_split_root']

    for section in dataset_config.sections():
        config = dataset_config[section]

        ref_path = config['ref_path']
        fast5_root = config['fast5_root']
        ref_root = config['ref_root']
        test_size = float(config.get('test_size', -1))
        sam_path = config['sam_path']

        if not only_split:
            logging.debug("Started %s" % section)
            logging.debug("Constructing .ref")
            construct_ref_files(fast5_root, sam_path, ref_root, ref_path)

        logging.debug("Started train-test split")
        # global train-test dest for all genomes
        reads_train_test_split(ref_root, test_size, ref_path)
        _make_train_test_symlinks(train_root, test_root, fast5_root, ref_root)

        # train-test dest per genome
        logging.debug("Started train-test split per genome")
        root = os.path.join(per_genome_split_root, section)
        genome_train_root = os.path.join(root, 'train')
        genome_test_root = os.path.join(root, 'test')
        _make_train_test_symlinks(genome_train_root, genome_test_root, fast5_root, ref_root)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("dataset_conf_path", help="path to dataset conf file defining \
                                                  all references used to construct dataset", type=str)
    parser.add_argument("--split", action='store_true', help="Only split train and tests set,\
                                        ref data should be already preprocessed")
    args = parser.parse_args()
    preprocess_all_ref(args.dataset_conf_path, args.split)

