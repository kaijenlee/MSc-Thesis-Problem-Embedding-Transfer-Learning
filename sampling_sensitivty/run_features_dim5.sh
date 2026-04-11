source ../../venv/bin/activate

cat << 'EOF' | xargs -P 10 -I {} bash -c "python generate_features_compressed.py {}"
--feature-type ela --sampling-method cma_random --dimension 5 --sample-size 10 --data-dir ../../data/samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method cma_random --dimension 5 --sample-size 25 --data-dir ../../data/samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method cma_random --dimension 5 --sample-size 50 --data-dir ../../data/samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method cma_random --dimension 5 --sample-size 75 --data-dir ../../data/samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method cma_random --dimension 5 --sample-size 100 --data-dir ../../data/samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method uniform --dimension 5 --sample-size 10 --data-dir ../../data/samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method uniform --dimension 5 --sample-size 25 --data-dir ../../data/samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method uniform --dimension 5 --sample-size 50 --data-dir ../../data/samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method uniform --dimension 5 --sample-size 75 --data-dir ../../data/samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method uniform --dimension 5 --sample-size 100 --data-dir ../../data/samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method lhs --dimension 5 --sample-size 10 --data-dir ../../data/samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method lhs --dimension 5 --sample-size 25 --data-dir ../../data/samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method lhs --dimension 5 --sample-size 50 --data-dir ../../data/samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method lhs --dimension 5 --sample-size 75 --data-dir ../../data/samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method lhs --dimension 5 --sample-size 100 --data-dir ../../data/samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method ilhs --dimension 5 --sample-size 10 --data-dir ../../data/samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method ilhs --dimension 5 --sample-size 25 --data-dir ../../data/samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method ilhs --dimension 5 --sample-size 50 --data-dir ../../data/samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method ilhs --dimension 5 --sample-size 75 --data-dir ../../data/samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method ilhs --dimension 5 --sample-size 100 --data-dir ../../data/samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method sobol --dimension 5 --sample-size 10 --data-dir ../../data/samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method sobol --dimension 5 --sample-size 25 --data-dir ../../data/samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method sobol --dimension 5 --sample-size 50 --data-dir ../../data/samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method sobol --dimension 5 --sample-size 75 --data-dir ../../data/samples --output-dir ~/constellation-kaijen/data/dim5featELA
--feature-type ela --sampling-method sobol --dimension 5 --sample-size 100 --data-dir ../../data/samples --output-dir ~/constellation-kaijen/data/dim5featELA
EOF



#--feature-type tla --sampling-method cma_random --sample-size 10 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method cma_random --sample-size 25 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method cma_random --sample-size 50 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method cma_random --sample-size 75 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method uniform --sample-size 75 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method uniform --sample-size 100 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method sobol --sample-size 100 --data-dir ../../data/samples/pickles



#--feature-type tla --sampling-method cma --sample-size 10 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method cma --sample-size 25 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method cma_random --sample-size 10 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method cma_random --sample-size 25 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method uniform --sample-size 10 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method uniform --sample-size 25 --data-dir ../../data/samples/pickles
#--feature-type ela --sampling-method cma --sample-size 25 --data-dir ../../data/samples/pickles
#--feature-type ela --sampling-method cma --sample-size 50 --data-dir ../../data/samples/pickles
#--feature-type ela --sampling-method cma_random --sample-size 25 --data-dir ../../data/samples/pickles
#--feature-type ela --sampling-method cma_random --sample-size 50 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method sobol --sample-size 10 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method sobol --sample-size 25 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method lhs --sample-size 10 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method lhs --sample-size 25 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method cma --sample-size 50 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method cma_random --sample-size 50 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method ilhs --sample-size 10 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method ilhs --sample-size 25 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method lhs --sample-size 50 --data-dir ../../data/samples/pickles
#--feature-type ela --sampling-method lhs --sample-size 50 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method ilhs --sample-size 50 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method sobol --sample-size 50 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method uniform --sample-size 50 --data-dir ../../data/samples/pickles
#--feature-type ela --sampling-method lhs --sample-size 75 --data-dir ../../data/samples/pickles
#--feature-type ela --sampling-method lhs --sample-size 100 --data-dir ../../data/samples/pickles
#--feature-type ela --sampling-method ilhs --sample-size 50 --data-dir ../../data/samples/pickles
#--feature-type ela --sampling-method uniform --sample-size 50 --data-dir ../../data/samples/pickles
#--feature-type ela --sampling-method uniform --sample-size 75 --data-dir ../../data/samples/pickles
#--feature-type ela --sampling-method uniform --sample-size 100 --data-dir ../../data/samples/pickles
#--feature-type ela --sampling-method sobol --sample-size 50 --data-dir ../../data/samples/pickles
#--feature-type ela --sampling-method sobol --sample-size 75 --data-dir ../../data/samples/pickles
#--feature-type ela --sampling-method sobol --sample-size 100 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method ilhs --sample-size 75 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method ilhs --sample-size 100 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method cma --sample-size 75 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method cma --sample-size 100 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method cma_random --sample-size 75 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method cma_random --sample-size 100 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method lhs --sample-size 75 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method lhs --sample-size 100 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method uniform --sample-size 75 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method uniform --sample-size 100 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method sobol --sample-size 75 --data-dir ../../data/samples/pickles
#--feature-type tla --sampling-method sobol --sample-size 100 --data-dir ../../data/samples/pickles
#--feature-type ela --sampling-method cma --sample-size 75 --data-dir ../../data/samples/pickles
#--feature-type ela --sampling-method cma --sample-size 100 --data-dir ../../data/samples/pickles
#--feature-type ela --sampling-method cma_random --sample-size 75 --data-dir ../../data/samples/pickles
#--feature-type ela --sampling-method cma_random --sample-size 100 --data-dir ../../data/samples/pickles
