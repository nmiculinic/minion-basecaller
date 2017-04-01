from dotenv import load_dotenv, find_dotenv
import model_utils
import os
load_dotenv(find_dotenv())


if __name__ == "__main__":
    model = model_utils.Ensamble(
        ['residual_deep'],
        [os.path.join(model_utils.repo_root, 'log', 'karla_docker', 'residual_deep_prod_17466_%d' % x) for x in [9939286]],
    )

    model.init_session()
    model.run_validation_full(frac=500)
    model.close_session()
