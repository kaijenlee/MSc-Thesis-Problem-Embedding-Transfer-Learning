source ../../venv/bin/activate

cat << 'EOF' | xargs -P 5 -I {} bash -c "python generate_features_compressed.py {}"
--feature-type ela --sampling-method ilhs --dimension 2 --sample-size 10 --data-dir ~/constellation-kaijen/samples --output-dir ~/constellation-kaijen/data/ilhsELA/dim2/
--feature-type ela --sampling-method ilhs --dimension 2 --sample-size 25 --data-dir ~/constellation-kaijen/samples --output-dir ~/constellation-kaijen/data/ilhsELA/dim2/
--feature-type ela --sampling-method ilhs --dimension 2 --sample-size 50 --data-dir ~/constellation-kaijen/samples --output-dir ~/constellation-kaijen/data/ilhsELA/dim2/
--feature-type ela --sampling-method ilhs --dimension 2 --sample-size 75 --data-dir ~/constellation-kaijen/samples --output-dir ~/constellation-kaijen/data/ilhsELA/dim2/
--feature-type ela --sampling-method ilhs --dimension 2 --sample-size 100 --data-dir ~/constellation-kaijen/samples --output-dir ~/constellation-kaijen/data/ilhsELA/dim2/
--feature-type ela --sampling-method ilhs --dimension 5 --sample-size 10 --data-dir ~/constellation-kaijen/samples --output-dir ~/constellation-kaijen/data/ilhsELA/dim5/
--feature-type ela --sampling-method ilhs --dimension 5 --sample-size 25 --data-dir ~/constellation-kaijen/samples --output-dir ~/constellation-kaijen/data/ilhsELA/dim5/
--feature-type ela --sampling-method ilhs --dimension 5 --sample-size 50 --data-dir ~/constellation-kaijen/samples --output-dir ~/constellation-kaijen/data/ilhsELA/dim5/
--feature-type ela --sampling-method ilhs --dimension 5 --sample-size 75 --data-dir ~/constellation-kaijen/samples --output-dir ~/constellation-kaijen/data/ilhsELA/dim5/
--feature-type ela --sampling-method ilhs --dimension 5 --sample-size 100 --data-dir ~/constellation-kaijen/samples --output-dir ~/constellation-kaijen/data/ilhsELA/dim5/
--feature-type ela --sampling-method ilhs --dimension 10 --sample-size 10 --data-dir ~/constellation-kaijen/samples --output-dir ~/constellation-kaijen/data/ilhsELA/dim2/
--feature-type ela --sampling-method ilhs --dimension 10 --sample-size 25 --data-dir ~/constellation-kaijen/samples --output-dir ~/constellation-kaijen/data/ilhsELA/dim2/
--feature-type ela --sampling-method ilhs --dimension 10 --sample-size 50 --data-dir ~/constellation-kaijen/samples --output-dir ~/constellation-kaijen/data/ilhsELA/dim2/
--feature-type ela --sampling-method ilhs --dimension 10 --sample-size 75 --data-dir ~/constellation-kaijen/samples --output-dir ~/constellation-kaijen/data/ilhsELA/dim2/
--feature-type ela --sampling-method ilhs --dimension 10 --sample-size 100 --data-dir ~/constellation-kaijen/samples --output-dir ~/constellation-kaijen/data/ilhsELA/dim2/
EOF
