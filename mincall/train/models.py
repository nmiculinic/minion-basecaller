import os
import logging
from typing import *

from keras import models, layers, regularizers, constraints, backend as K
from mincall.common import *
from ._input_feeders import DataQueue
from mincall.train import ops
from minion_data import dataset_pb2
import tensorflow as tf
from ._types import *
from .layers import *
from glob import glob
import voluptuous

logger = logging.getLogger(__name__)


class AbstractModel:
    cfg_class: Any
    forward_model: models.Model
    ratio: int
    autoencoder_model: Optional[models.Model] = None
    autoenc_coeff: float = 1.0

    def __init__(self, forward_model, ratio, autoencoder_model=None, autoenc_coeff=1):
        self.ratio = ratio
        self.forward_model = forward_model
        self.autoencoder_model = autoencoder_model
        self.autoenc_coeff = autoenc_coeff

        with K.name_scope("export"):
            self.x = tf.placeholder(tf.float32, shape=(None, None, 1))
            self.y = self.forward_model(self.x)
            print(self.y.shape)


    def bind(self, cfg: InputFeederCfg,
             data_dir: List[DataDir]) -> 'BindedModel':
        assert cfg.ratio == self.ratio
        return BindedModel(
            cfg=cfg,
            data_dir=data_dir,
            forward_model=self.forward_model,
            autoencoder_model=self.autoencoder_model,
            autoenc_coeff=self.autoenc_coeff,
        )

    def save(self, sess: tf.Session, folder: str, step: int):
        p = os.path.join(folder, f"full-model-{step:05}.save")
        self.forward_model.save(p, overwrite=True, include_optimizer=False)

        model_input = tf.saved_model.utils.build_tensor_info(self.x)
        model_output = tf.saved_model.utils.build_tensor_info(self.y)

        signature_definition = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={"x": model_input},
            outputs={"y": model_output},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )

        export_path = os.path.join(folder, "saver", "1")
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                "mincall":
                    signature_definition,
            },
        )
        builder.save()

class BindedModel:
    """Class representing model binded to the dataset with all required properties.

    It shouldn't be created direcly by the end consumer
    """

    def __init__(
        self,
        cfg: InputFeederCfg,
        data_dir: List[DataDir],
        forward_model: models.Model,
        autoencoder_model: models.Model = None,
        autoenc_coeff: float = 1,
    ):
        self.dataset = []
        for x in data_dir:
            dps = list(glob(f"{x.dir}/*.datapoint"))
            self.dataset.extend(dps)
            logger.info(
                f"Added {len(dps)} datapoint from {x.name} to train set; dir: {x.dir}"
            )

        self._logger = logging.getLogger(__name__)
        self.learning_phase = K.learning_phase()
        with K.name_scope("data_in"):
            self.dq = DataQueue(
                cfg,
                self.dataset,
                capacity=10 * cfg.batch_size,
                min_after_deque=2 * cfg.batch_size
            )
        input_signal: tf.Tensor = self.dq.batch_signal
        input_signal = tf.Print(
            input_signal, [tf.shape(input_signal)],
            first_n=1,
            summarize=10,
            message="input signal shape, [batch_size,max_time, 1]"
        )

        labels: tf.SparseTensor = self.dq.batch_labels
        signal_len: tf.Tensor = self.dq.batch_signal_len

        self.labels = labels

        untransposed_logits = forward_model(input_signal)
        self.logits = tf.transpose(untransposed_logits, [1, 0, 2]
                                  )  # [max_time, batch_size, class_num]
        self._logger.info(f"Logits shape: {self.logits.shape}")
        self.logits = tf.Print(
            self.logits, [tf.shape(self.logits)],
            first_n=1,
            summarize=10,
            message="logits shape [max_time, batch_size, class_num]"
        )

        seq_len = tf.cast(
            tf.floor_div(signal_len + cfg.ratio - 1, cfg.ratio), tf.int32
        )  # Round up
        seq_len = tf.Print(
            seq_len, [tf.shape(seq_len), seq_len],
            first_n=5,
            summarize=15,
            message="seq_len [expected around max_time]"
        )

        self.ctc_loss_unaggregated = tf.nn.ctc_loss(
            labels=labels,
            inputs=self.logits,
            sequence_length=seq_len,
            preprocess_collapse_repeated=False,
            ctc_merge_repeated=True,
            time_major=True,
        )

        self.predict = tf.nn.ctc_beam_search_decoder(
            inputs=self.logits,
            sequence_length=seq_len,
            merge_repeated=cfg.
            surrogate_base_pair,  # Gotta merge if we have surrogate_base_pairs
            top_paths=1,
            beam_width=100,
        )[0][0]

        finite_mask = tf.logical_not(
            tf.logical_or(
                tf.is_nan(self.ctc_loss_unaggregated),
                tf.is_inf(self.ctc_loss_unaggregated),
            )
        )

        # self.ctc_loss = tf.reduce_mean(self.losses)
        self.ctc_loss = tf.reduce_mean(
            tf.boolean_mask(
                self.ctc_loss_unaggregated,
                finite_mask,
            )
        )
        if forward_model.losses:
            self.regularization_loss = tf.add_n(forward_model.losses)
        else:
            self.regularization_loss = tf.constant(0.0)
        # Nice little hack to get inf/NaNs out of the way. In the beginning of the training
        # logits shall move to some unrealistically large numbers and it shall be hard
        # finding path through the network
        self.regularization_loss += tf.train.exponential_decay(
            learning_rate=tf.nn.l2_loss(self.logits),
            global_step=tf.train.get_or_create_global_step(),
            decay_rate=0.5,
            decay_steps=200,
        )

        self.total_loss = [self.ctc_loss, self.regularization_loss]

        percent_finite = tf.reduce_mean(tf.cast(finite_mask, tf.int32))
        percent_finite = tf.Print(
            percent_finite, [percent_finite], first_n=10, message="%finite"
        )
        self.summaries = []
        self.ext_summaries = []

        if autoencoder_model is not None:
            if autoencoder_model.losses:
                regularization_loss = tf.reduce_mean(
                    tf.add_n(autoencoder_model.losses)
                )
            else:
                regularization_loss = tf.constant(0.0)
            self.summaries.append(
                tf.summary.scalar(
                    'regularization_loss_autoenc',
                    regularization_loss,
                    family="losses"
                )
            )
            self.total_loss.append(regularization_loss)

            signal_reconstruction = autoencoder_model(untransposed_logits)
            autoencoder_loss = autoenc_coeff * ops.autoencoder_loss(
                signal=input_signal,
                signal_reconstruction=signal_reconstruction,
                signal_len=signal_len,
            )

            self.summaries.append(
                tf.summary.scalar(
                    'autoenc_loss', autoencoder_loss, family="losses"
                )
            )
            self.ext_summaries.extend(
                tensor_default_summaries(
                    "autoenc_signal", signal_reconstruction, family="signal"
                ),
            )
            self.total_loss.append(autoencoder_loss)

        self.total_loss = tf.add_n(self.total_loss)
        self.summaries.extend([
            tf.summary.scalar(f'total_loss', self.total_loss, family="losses"),
            tf.summary.scalar(f'ctc_loss', self.ctc_loss, family="losses"),
            tf.summary.scalar(
                f'regularization_loss_forward',
                self.regularization_loss,
                family="losses"
            ),
            tf.summary.scalar("finite_percent", percent_finite),
            *self.dq.summaries,
        ])

        self.ext_summaries.extend(
            tensor_default_summaries(
                "input_signal", input_signal, family="signal"
            ),
        )

        *self.alignment_stats, self.identity = tf.py_func(
            ops.alignment_stats,
            [
                self.labels.indices, self.labels.values, self.predict.indices,
                self.predict.values, self.labels.dense_shape[0]
            ],
            (len(ops.aligment_stats_ordering) + 1) * [tf.float32],
            stateful=False,
        )

        for stat_type, stat in zip(
            ops.aligment_stats_ordering, self.alignment_stats
        ):
            stat.set_shape((None,))
            self.ext_summaries.append(
                tensor_default_summaries(
                    dataset_pb2.Cigar.Name(stat_type) + "_rate",
                    stat,
                )
            )

        self.identity.set_shape((None,))
        self.ext_summaries.append(
            tensor_default_summaries(
                "IDENTITY",
                self.identity,
            )
        )
        self.ext_summaries.extend(
            tensor_default_summaries("logits", self.logits, family="logits")
        )

        self.ext_summaries.append(
            tf.summary.image(
                "logits",
                tf.expand_dims(
                    tf.nn.softmax(tf.transpose(self.logits, [1, 2, 0])),
                    -1,
                )
            )
        )
        self.ext_summaries.extend(self.summaries)

    def input_wrapper(self, sess: tf.Session, coord: tf.train.Coordinator):
        return self.dq.start_input_processes(sess, coord)



class DummyCfg(NamedTuple):
    num_layers: int

    @classmethod
    def scheme(cls, data):
        return named_tuple_helper(cls, {}, data)


class DummyModel(AbstractModel):
    cfg_class = DummyCfg

    def __init__(self, n_classes, hparams: Dict):
        cfg: DummyCfg = DummyCfg.scheme(hparams)
        super().__init__(
            forward_model=self._foraward_model(n_classes, cfg),
            ratio= 1,
            autoencoder_model=self._backwards(n_classes, cfg),
        )


    @staticmethod
    def _foraward_model(n_classes, cfg: DummyCfg) -> models.Model:
        input = layers.Input(shape=(None, 1))
        net = input
        for _ in range(cfg.num_layers):
            net = layers.BatchNormalization()(net)
            net = layers.Conv1D(
                10,
                3,
                padding="same",
                dilation_rate=2,
                bias_regularizer=regularizers.l1(0.1)
            )(net)
            net = layers.Activation('relu')(net)

        net = layers.Conv1D(n_classes, 3, padding="same")(net)
        return models.Model(inputs=[input], outputs=[net])

    @staticmethod
    def _backwards(n_classes, cfg: DummyCfg) -> models.Model:
        input = layers.Input(shape=(None, n_classes))
        net = input
        for _ in range(cfg.num_layers):
            net = layers.BatchNormalization()(net)
            net = layers.Conv1D(
                10,
                3,
                padding="same",
                dilation_rate=2,
                bias_regularizer=regularizers.l1(0.1)
            )(net)
            net = layers.Activation('relu')(net)

        net = layers.Conv1D(1, 3, padding="same")(net)
        return models.Model(inputs=[input], outputs=[net])


class Big01Cfg(NamedTuple):
    num_blocks: int
    block_elem: int
    block_init_channels: int = 32
    receptive_width: int = 5

    @classmethod
    def scheme(cls, data):
        return named_tuple_helper(cls, {}, data)


class Big01(AbstractModel):
    cfg_class = Big01Cfg

    def __init__(self, n_classes: int, hparams: Dict):
        cfg: Big01Cfg = Big01Cfg.scheme(hparams)
        input = layers.Input(shape=(None, 1))
        net = input
        for i in range(cfg.num_blocks):
            channels = 2**i * cfg.block_init_channels
            net = layers.Conv1D(
                channels,
                cfg.receptive_width,
                padding="same",
                bias_regularizer=regularizers.l1(0.1)
            )(net)
            with tf.name_scope(f"block_{i}"):
                for _ in range(cfg.block_elem):
                    x = net
                    net = layers.Conv1D(
                        channels, cfg.receptive_width, padding='same'
                    )(net)
                    net = layers.BatchNormalization()(net)
                    net = layers.Activation('relu')(net)
                    net = layers.Conv1D(
                        channels, cfg.receptive_width, padding='same'
                    )(net)
                    net = layers.BatchNormalization()(net)
                    net = layers.Activation('relu')(net)
                    net = ConstMultiplierLayer()(net)
                    net = layers.add([x, net])
            net = layers.MaxPool1D(padding='same', pool_size=2)(net)

        net = layers.Conv1D(n_classes, cfg.receptive_width, padding="same")(net)
        self.forward_model = models.Model(inputs=[input], outputs=[net])
        self.ratio = 2**cfg.num_blocks


all_models: Dict[str, Callable[[str], AbstractModel]] = {
    'dummy': DummyModel,
    'big_01': Big01,
}
