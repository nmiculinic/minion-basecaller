import model_utils
import os
model = model_utils.Ensamble(
    ['model_small'],
    [os.path.join(model_utils.repo_root, 'log', 'protagonist', 'model_small_18071_9943114')],
)

model.init_session()
m1 = model.basecall_sample('/home/lpp/Downloads/minion/pass/26075_ch161_read656_strand.fast5')
model.close_session()
m1_ld = model.log_dir

import eval
model = eval.load_model('model_small', os.path.join(model_utils.repo_root, 'log', 'protagonist', 'model_small_18071_9943114'))
model.init_session()
model.restore()
m2 = model.basecall_sample('/home/lpp/Downloads/minion/pass/26075_ch161_read656_strand.fast5')
model.close_session()
print(m1, m1_ld)
print(m2)
