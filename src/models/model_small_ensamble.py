from dotenv import load_dotenv, find_dotenv
import model_utils
import os
load_dotenv(find_dotenv())


if __name__ == "__main__":
    model = model_utils.Ensamble(
        ['model_small'],
        [os.path.join(model_utils.repo_root, 'log', 'protagonist', 'model_small_18071_9943114')],
    )

    model.init_session()
    model.run_validation_full(frac=500)
    model.close_session()
