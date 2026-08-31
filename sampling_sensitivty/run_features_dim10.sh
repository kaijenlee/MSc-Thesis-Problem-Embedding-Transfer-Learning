source ../../venv/bin/activate

cat << 'EOF' | xargs -P 10 -I {} bash -c "python generate_features_compressed.py {}"
--feature-type ela --sampling-method cma_random --dimension 10 --sample-size 25 --data-dir ../../new_samples--output-dir ~/constellation-kaijen/data/dim10featELA
--feature-type ela --sampling-method cma_random --dimension 10 --sample-size 50 --data-dir ../../new_samples--output-dir ~/constellation-kaijen/data/dim10featELA
--feature-type ela --sampling-method cma_random --dimension 10 --sample-size 75 --data-dir ../../new_samples--output-dir ~/constellation-kaijen/data/dim10featELA
--feature-type ela --sampling-method cma_random --dimension 10 --sample-size 100 --data-dir ../../new_samples--output-dir ~/constellation-kaijen/data/dim10featELA
--feature-type ela --sampling-method uniform --dimension 10 --sample-size 25 --data-dir ../../new_samples--output-dir ~/constellation-kaijen/data/dim10featELA
--feature-type ela --sampling-method uniform --dimension 10 --sample-size 50 --data-dir ../../new_samples--output-dir ~/constellation-kaijen/data/dim10featELA
--feature-type ela --sampling-method uniform --dimension 10 --sample-size 75 --data-dir ../../new_samples--output-dir ~/constellation-kaijen/data/dim10featELA
--feature-type ela --sampling-method uniform --dimension 10 --sample-size 100 --data-dir ../../new_samples--output-dir ~/constellation-kaijen/data/dim10featELA
--feature-type ela --sampling-method lhs --dimension 10 --sample-size 25 --data-dir ../../new_samples--output-dir ~/constellation-kaijen/data/dim10featELA
--feature-type ela --sampling-method lhs --dimension 10 --sample-size 50 --data-dir ../../new_samples--output-dir ~/constellation-kaijen/data/dim10featELA
--feature-type ela --sampling-method lhs --dimension 10 --sample-size 75 --data-dir ../../new_samples--output-dir ~/constellation-kaijen/data/dim10featELA
--feature-type ela --sampling-method lhs --dimension 10 --sample-size 100 --data-dir ../../new_samples--output-dir ~/constellation-kaijen/data/dim10featELA
--feature-type ela --sampling-method ilhs --dimension 10 --sample-size 25 --data-dir ../../new_samples--output-dir ~/constellation-kaijen/data/dim10featELA
--feature-type ela --sampling-method ilhs --dimension 10 --sample-size 50 --data-dir ../../new_samples--output-dir ~/constellation-kaijen/data/dim10featELA
--feature-type ela --sampling-method ilhs --dimension 10 --sample-size 75 --data-dir ../../new_samples--output-dir ~/constellation-kaijen/data/dim10featELA
--feature-type ela --sampling-method ilhs --dimension 10 --sample-size 100 --data-dir ../../new_samples--output-dir ~/constellation-kaijen/data/dim10featELA
--feature-type ela --sampling-method lhs_random_cd --dimension 10 --sample-size 25 --data-dir ../../new_samples--output-dir ~/constellation-kaijen/data/dim10featELA
--feature-type ela --sampling-method lhs_random_cd --dimension 10 --sample-size 50 --data-dir ../../new_samples--output-dir ~/constellation-kaijen/data/dim10featELA
--feature-type ela --sampling-method lhs_random_cd --dimension 10 --sample-size 75 --data-dir ../../new_samples--output-dir ~/constellation-kaijen/data/dim10featELA
--feature-type ela --sampling-method lhs_random_cd --dimension 10 --sample-size 100 --data-dir ../../new_samples--output-dir ~/constellation-kaijen/data/dim10featELA
--feature-type ela --sampling-method sobol --dimension 10 --sample-size 25 --data-dir ../../new_samples--output-dir ~/constellation-kaijen/data/dim10featELA
--feature-type ela --sampling-method sobol --dimension 10 --sample-size 50 --data-dir ../../new_samples--output-dir ~/constellation-kaijen/data/dim10featELA
--feature-type ela --sampling-method sobol --dimension 10 --sample-size 75 --data-dir ../../new_samples--output-dir ~/constellation-kaijen/data/dim10featELA
--feature-type ela --sampling-method sobol --dimension 10 --sample-size 100 --data-dir ../../new_samples--output-dir ~/constellation-kaijen/data/dim10featELA
EOF
