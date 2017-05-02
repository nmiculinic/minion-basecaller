import glob
import logging
import os
import shutil
import subprocess
import tempfile
import time
from argparse import ArgumentParser
from configparser import ConfigParser

import h5py
import pysam
from tqdm import tqdm

from mincall.bioinf_utils import reverse_complement, decompress_cigar_pairs, read_fasta


def get_target_sequences(sam_out):
    result_dict = {}

    with pysam.AlignmentFile(sam_out, "r") as samfile:
        for x in samfile.fetch():
            name = x.query_name

            if x.is_unmapped:
                logging.warning("%s unmapped" % name)
                continue
            try:
                target = x.get_reference_sequence()
            except ValueError:
                logging.error("%s Mapped but reference len equals 0" % name)
                continue

            ref_name = x.reference_name
            length = x.reference_length
            start_pos = x.reference_start
            cigar_pairs = x.cigartuples

            if x.is_reverse:
                target = reverse_complement(target)
                cigar_pairs = list(reversed(cigar_pairs))

            cigar_str = decompress_cigar_pairs(cigar_pairs, mode='ints')
            result_dict[name] = [target, ref_name, start_pos, length, cigar_str]
    return result_dict


def _align_for_reference_batch(files_in_batch, generate_sam_f, ref_starts, out_root):
    name_to_file = {}
    tmp_work_dir = tempfile.mkdtemp()
    tmp_sam_path = os.path.join(tmp_work_dir, 'tmp.sam')
    tmp_fastq_path = os.path.join(tmp_work_dir, 'tmp.fastq')
    s_load = time.time()
    with open(tmp_fastq_path, 'wb') as fq_file:
        for f in files_in_batch:
            with h5py.File(f, 'r') as h5:
                fastq = h5['/Analyses/Basecall_1D_000/BaseCalled_template/Fastq'][()]
                read_name, *_ = fastq.strip().split(b'\n')
                read_name = read_name[1:].split(b' ')[0].decode()
                assert read_name not in name_to_file
                name_to_file[read_name] = os.path.basename(f)

                fq_file.write(fastq)
                fq_file.write(b'\n')
        logging.debug("load fq", time.time() - s_load)
        s_sam = time.time()
        generate_sam_f(tmp_fastq_path, tmp_sam_path)
        logging.debug("generate sam", time.time() - s_sam)
        s_dict = time.time()
        result_dict = get_target_sequences(tmp_sam_path)
        logging.debug("result_dict", time.time() - s_dict)

    # cleanup
    shutil.rmtree(tmp_work_dir)
    s_dump_files = time.time()
    for name, (target, ref_name, start_position, length, cigar) in result_dict.items():
        basename, ext = os.path.splitext(name_to_file[name])
        ref_out_name = basename + '.ref'
        out_ref_path = os.path.join(out_root, ref_out_name)

        abs_start_pos = ref_starts[ref_name] + start_position

        with open(out_ref_path, 'w') as fout:
            fout.write('%s\t%s\n' % (name, ref_name))
            fout.write('%d\t%d\t%d\n' % (abs_start_pos, start_position, length))
            fout.write(cigar + '\n')
            fout.write(target + '\n')
    print("dump_all", time.time() - s_dump_files)
    print("total", time.time() - s_load)


def generate_sam_graphmap(ref_path, is_circular):
    def _generate_sam(fastq_path, sam_path):
        args = ["graphmap", "align", "-r", ref_path, "-d", fastq_path, "-o", sam_path, "-v", "0", "--extcigar"]
        if is_circular:
            args.append("-C")

        exit_status = subprocess.call(args)
        if exit_status != 0:
            logging.warning("Graphmap exit status %d" % exit_status)

    return _generate_sam


def get_ref_starts_dict(fa_path):
    start_position = {}
    current_len = 0
    with pysam.FastxFile(fa_path) as fh:
        for entry in fh:
            start_position[entry.name] = current_len
            current_len += len(entry.sequence)
    return start_position


def reads_train_test_split(ref_root, test_size, ref_path):
    reference_len = len(read_fasta(ref_path))
    train_size = 1 - test_size

    files = glob.glob(os.path.join(ref_root, '*.ref'))
    train_path = os.path.join(ref_root, 'train.txt')
    test_path = os.path.join(ref_root, 'test.txt')

    with open(train_path, 'w') as trainf, open(test_path, 'w') as testf:
        for file_path in files:
            basename = os.path.basename(file_path)
            name, ext = os.path.splitext(basename)

            with open(file_path, 'r') as fin:
                next(fin)  # skip header
                line = next(fin)  # line2: abs_start_pos\tstart_position\tlength
                start, rel_start, length = [int(x) for x in line.split()]
                end = start + length
                if end < reference_len * train_size:
                    # train data
                    trainf.write("%s\t%d" % (name, length))

                elif start > reference_len * train_size and end <= reference_len:
                    # starts after 'split' and does not overlap in case of circular aligment
                    # test data
                    testf.write("%s\t%d" % (name, length))
                else:
                    logging.info('Skipping ref, overlaps train and test')


def align_for_reference(fast5_in, out_dir, generate_sam_f, ref_path, batch_size):
    if os.path.isfile(fast5_in):
        file_list = [fast5_in]
    elif os.path.isdir(fast5_in):
        file_list = glob.glob(os.path.join(fast5_in, '*.fast5'))
    else:
        logging.error("Invalid fastin - expected file or dir %s, skipping!!!" % fast5_in)
        return

    n_files = len(file_list)
    os.makedirs(out_dir, exist_ok=True)
    ref_starts = get_ref_starts_dict(ref_path)

    for i in tqdm(range(0, n_files, batch_size)):
        files_in_batch = file_list[i:i + batch_size]
        _align_for_reference_batch(files_in_batch, generate_sam_f, ref_starts, out_dir)


def preprocess_all_ref(dataset_conf_path, only_split, generate_sam_f):
    dataset_config = ConfigParser()
    dataset_config.read(dataset_conf_path)
    batch_size = int(dataset_config['DEFAULT']['batch_size'])

    for section in dataset_config.sections():
        config = dataset_config[section]

        ref_path = config['ref_path']
        fast5_root = config['fast5_root']
        ref_root = config['ref_root']
        is_circular = bool(config['circular'])
        test_size = float(config.get('test_size', -1))

        if not only_split:
            logging.info("Started %s" % section)
            generate_sam_f = generate_sam_f(ref_path, is_circular)
            logging.info("Started aligning")
            align_for_reference(fast5_root, ref_root, generate_sam_f, ref_path, batch_size)

        logging.info("Started train-test split")
        reads_train_test_split(ref_root, test_size, ref_path)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("dataset_conf_path", help="path to dataset conf file defining \
                                                  all references used to construct dataset", type=str)
    parser.add_argument("--split", help="Only split train and tests set,\
                                        ref data should be already preprocessed")
    args = parser.parse_args()

    preprocess_all_ref(args.dataset_conf_path, args.split, generate_sam_graphmap)
