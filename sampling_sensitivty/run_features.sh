source ../../venv/bin/activate

cat << 'EOF' | xargs -P 8 -I {} bash -c "python generate_features_compressed.py {}"
--feature-type ela --sampling-method cma_random --sample-size 10 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type ela --sampling-method cma_random --sample-size 25 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type ela --sampling-method cma_random --sample-size 50 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type ela --sampling-method cma_random --sample-size 75 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type ela --sampling-method cma_random --sample-size 100 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type ela --sampling-method lhs --sample-size 10 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type ela --sampling-method lhs --sample-size 25 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type ela --sampling-method lhs --sample-size 75 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type ela --sampling-method lhs --sample-size 50 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type ela --sampling-method lhs --sample-size 100 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type ela --sampling-method ilhs --sample-size 10 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type ela --sampling-method ilhs --sample-size 25 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type ela --sampling-method ilhs --sample-size 75 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type ela --sampling-method ilhs --sample-size 50 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type ela --sampling-method ilhs --sample-size 100 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type ela --sampling-method uniform --sample-size 10 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type ela --sampling-method uniform --sample-size 50 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type ela --sampling-method uniform --sample-size 25 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type ela --sampling-method uniform --sample-size 75 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type ela --sampling-method uniform --sample-size 100 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type ela --sampling-method sobol --sample-size 10 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type ela --sampling-method sobol --sample-size 25 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type ela --sampling-method sobol --sample-size 50 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type ela --sampling-method sobol --sample-size 75 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type ela --sampling-method sobol --sample-size 100 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type tla --sampling-method cma_random --sample-size 10 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type tla --sampling-method cma_random --sample-size 25 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type tla --sampling-method cma_random --sample-size 50 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type tla --sampling-method cma_random --sample-size 75 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type tla --sampling-method cma_random --sample-size 100 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type tla --sampling-method lhs --sample-size 10 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type tla --sampling-method lhs --sample-size 25 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type tla --sampling-method lhs --sample-size 75 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type tla --sampling-method lhs --sample-size 50 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type tla --sampling-method lhs --sample-size 100 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type tla --sampling-method ilhs --sample-size 10 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type tla --sampling-method ilhs --sample-size 25 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type tla --sampling-method ilhs --sample-size 75 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type tla --sampling-method ilhs --sample-size 50 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type tla --sampling-method ilhs --sample-size 100 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type tla --sampling-method uniform --sample-size 10 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type tla --sampling-method uniform --sample-size 50 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type tla --sampling-method uniform --sample-size 25 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type tla --sampling-method uniform --sample-size 75 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type tla --sampling-method uniform --sample-size 100 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type tla --sampling-method sobol --sample-size 10 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type tla --sampling-method sobol --sample-size 25 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type tla --sampling-method sobol --sample-size 50 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type tla --sampling-method sobol --sample-size 75 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
--feature-type tla --sampling-method sobol --sample-size 100 --data-dir ../../data/samples/pickles --output-dir ~/constellation-kaijen/data/dim3feat
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
