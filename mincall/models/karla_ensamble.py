from dotenv import load_dotenv, find_dotenv
import model_utils
import os
load_dotenv(find_dotenv())

models = [
    ('model_big_1', 'model_big_1'),
    ('model_big_2', 'model_big_2'),
    ('residual_deep', 'karla_docker/residual_deep_prod_17466_9939286'),
    ('residual_deep', 'karla_docker/residual_deep_prod_17466_9895025'),
    ('residual_deep', 'karla_docker/residual_deep_prod_17466_2'),
]

if __name__ == "__main__":
    model = model_utils.Ensamble(
        [x[0] for x in models],
        [os.path.join(model_utils.repo_root, 'log', x[1]) for x in models],
    )

    model.init_session()
    model.run_validation_full(frac=500)
    model.close_session()
