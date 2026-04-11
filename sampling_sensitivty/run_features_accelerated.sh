source ../../venv/bin/activate

cat << 'EOF' | xargs -P 1 -I {} bash -c "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python generate_features_compressed.py {}"
--feature-type tla --sampling-method cma_random --dimension 2 --sample-size 75 --data-dir ../../data/samples --output-dir ~/constellation-kaijen/data/dim2featTLA --n-jobs 10 --use-ripser-plus-plus
--feature-type tla --sampling-method cma_random --dimension 2 --sample-size 100 --data-dir ../../data/samples --output-dir ~/constellation-kaijen/data/dim2featTLA --n-jobs 10 --use-ripser-plus-plus
--feature-type tla --sampling-method uniform --dimension 2 --sample-size 100 --data-dir ../../data/samples --output-dir ~/constellation-kaijen/data/dim2featTLA --n-jobs 10 --use-ripser-plus-plus
--feature-type tla --sampling-method lhs --dimension 2 --sample-size 100 --data-dir ../../data/samples --output-dir ~/constellation-kaijen/data/dim2featTLA --n-jobs 10 --use-ripser-plus-plus
--feature-type tla --sampling-method ilhs --dimension 2 --sample-size 100 --data-dir ../../data/samples --output-dir ~/constellation-kaijen/data/dim2featTLA --n-jobs 10 --use-ripser-plus-plus
--feature-type tla --sampling-method sobol --dimension 2 --sample-size 100 --data-dir ../../data/samples --output-dir ~/constellation-kaijen/data/dim2featTLA --n-jobs 10 --use-ripser-plus-plus
--feature-type tla --sampling-method cma --dimension 2 --sample-size 100 --data-dir ../../data/samples --output-dir ~/constellation-kaijen/data/dim2featTLA --n-jobs 10 --use-ripser-plus-plus
EOF