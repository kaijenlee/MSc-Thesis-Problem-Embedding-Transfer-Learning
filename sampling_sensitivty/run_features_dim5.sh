source ../../venv/bin/activate

cat << 'EOF' | xargs -P 5 -I {} bash -c "python generate_features_compressed.py {}"
--feature-type ela --sampling-method cma_random --dimension 5 --sample-size 25 --data-dir ../../new_samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method cma_random --dimension 5 --sample-size 50 --data-dir ../../new_samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method cma_random --dimension 5 --sample-size 75 --data-dir ../../new_samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method cma_random --dimension 5 --sample-size 100 --data-dir ../../new_samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method uniform --dimension 5 --sample-size 25 --data-dir ../../new_samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method uniform --dimension 5 --sample-size 50 --data-dir ../../new_samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method uniform --dimension 5 --sample-size 75 --data-dir ../../new_samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method uniform --dimension 5 --sample-size 100 --data-dir ../../new_samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method lhs --dimension 5 --sample-size 25 --data-dir ../../new_samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method lhs --dimension 5 --sample-size 50 --data-dir ../../new_samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method lhs --dimension 5 --sample-size 75 --data-dir ../../new_samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method lhs --dimension 5 --sample-size 100 --data-dir ../../new_samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method lhs_random_cd --dimension 5 --sample-size 25 --data-dir ../../new_samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method lhs_random_cd --dimension 5 --sample-size 50 --data-dir ../../new_samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method lhs_random_cd --dimension 5 --sample-size 75 --data-dir ../../new_samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method lhs_random_cd --dimension 5 --sample-size 100 --data-dir ../../new_samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method ilhs --dimension 5 --sample-size 25 --data-dir ../../new_samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method ilhs --dimension 5 --sample-size 50 --data-dir ../../new_samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method ilhs --dimension 5 --sample-size 75 --data-dir ../../new_samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method ilhs --dimension 5 --sample-size 100 --data-dir ../../new_samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method sobol --dimension 5 --sample-size 25 --data-dir ../../new_samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method sobol --dimension 5 --sample-size 50 --data-dir ../../new_samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method sobol --dimension 5 --sample-size 75 --data-dir ../../new_samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method sobol --dimension 5 --sample-size 100 --data-dir ../../new_samples --output-dir ~/constellation-kaijen/data/dim5featELA
EOF
