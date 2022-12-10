for i in 0 1 2 4 8; do
  res_dir=masks_$i
  CITYSCAPES_RESULTS=$res_dir CITYSCAPES_DATASET=../data python -m cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling
done
