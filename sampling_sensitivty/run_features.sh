source ../venv/bin/activate

cat << 'EOF' | xargs -P 6 -I {} bash -c "python generate_features.py {}"
--feature-type tla --sampling-method cma --sample-size 10 --data-dir ../../data/samples/pickles
--feature-type tla --sampling-method cma --sample-size 25 --data-dir ../../data/samples/pickles
--feature-type tla --sampling-method cma --sample-size 50 --data-dir ../../data/samples/pickles
--feature-type tla --sampling-method lhs --sample-size 10 --data-dir ../../data/samples/pickles
--feature-type tla --sampling-method lhs --sample-size 25 --data-dir ../../data/samples/pickles
--feature-type tla --sampling-method lhs --sample-size 50 --data-dir ../../data/samples/pickles
--feature-type tla --sampling-method ilhs --sample-size 10 --data-dir ../../data/samples/pickles
--feature-type tla --sampling-method ilhs --sample-size 25 --data-dir ../../data/samples/pickles
--feature-type tla --sampling-method ilhs --sample-size 50 --data-dir ../../data/samples/pickles
--feature-type tla --sampling-method uniform --sample-size 10 --data-dir ../../data/samples/pickles
--feature-type tla --sampling-method uniform --sample-size 25 --data-dir ../../data/samples/pickles
--feature-type tla --sampling-method uniform --sample-size 50 --data-dir ../../data/samples/pickles
--feature-type tla --sampling-method sobol --sample-size 10 --data-dir ../../data/samples/pickles
--feature-type tla --sampling-method sobol --sample-size 25 --data-dir ../../data/samples/pickles
--feature-type tla --sampling-method sobol --sample-size 50 --data-dir ../../data/samples/pickles
--feature-type ela --sampling-method cma --sample-size 10 --data-dir ../../data/samples/pickles
--feature-type ela --sampling-method cma --sample-size 25 --data-dir ../../data/samples/pickles
--feature-type ela --sampling-method cma --sample-size 50 --data-dir ../../data/samples/pickles
--feature-type ela --sampling-method lhs --sample-size 10 --data-dir ../../data/samples/pickles
--feature-type ela --sampling-method lhs --sample-size 25 --data-dir ../../data/samples/pickles
--feature-type ela --sampling-method lhs --sample-size 50 --data-dir ../../data/samples/pickles
--feature-type ela --sampling-method ilhs --sample-size 10 --data-dir ../../data/samples/pickles
--feature-type ela --sampling-method ilhs --sample-size 25 --data-dir ../../data/samples/pickles
--feature-type ela --sampling-method ilhs --sample-size 50 --data-dir ../../data/samples/pickles
--feature-type ela --sampling-method uniform --sample-size 10 --data-dir ../../data/samples/pickles
--feature-type ela --sampling-method uniform --sample-size 25 --data-dir ../../data/samples/pickles
--feature-type ela --sampling-method uniform --sample-size 50 --data-dir ../../data/samples/pickles
--feature-type ela --sampling-method sobol --sample-size 10 --data-dir ../../data/samples/pickles
--feature-type ela --sampling-method sobol --sample-size 25 --data-dir ../../data/samples/pickles
--feature-type ela --sampling-method sobol --sample-size 50 --data-dir ../../data/samples/pickles
EOF
