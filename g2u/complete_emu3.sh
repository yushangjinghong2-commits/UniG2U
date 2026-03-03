#!/bin/bash
# Emu3-Chat - Complete Evaluation Script
# Runs multiple understanding tasks with Emu3-Chat model

echo "======================================"
echo "Emu3-Chat Complete Evaluation"
echo "Running multiple tasks..."
echo "======================================"
echo ""

# ChartQA - Chart understanding
bash /home/aiscuser/lmms-eval/g2u/emu3.sh "0" "chartqa100" "./logs/emu3/chartqa" "BAAI/Emu3-Chat-hf" "29602"

# IllusionBench - Visual illusion understanding
bash /home/aiscuser/lmms-eval/g2u/emu3.sh "0" "illusionbench_arshia_test" "./logs/emu3/illusionbench" "BAAI/Emu3-Chat-hf" "29603"

# VisualPuzzles - Visual reasoning
bash /home/aiscuser/lmms-eval/g2u/emu3.sh "1" "VisualPuzzles" "./logs/emu3/visualpuzzles" "BAAI/Emu3-Chat-hf" "29604"

# RealUnify - Real-world understanding
bash /home/aiscuser/lmms-eval/g2u/emu3.sh "3" "realunify" "./logs/emu3/realunify" "BAAI/Emu3-Chat-hf" "29605"

# MMSI - Multimodal spatial intelligence
bash /home/aiscuser/lmms-eval/g2u/emu3.sh "2" "mmsi" "./logs/emu3/mmsi" "BAAI/Emu3-Chat-hf" "29606"

# Uni-MMMU - Multimodal understanding
bash /home/aiscuser/lmms-eval/g2u/emu3.sh "4" "uni_mmmu" "./logs/emu3/uni_mmmu" "BAAI/Emu3-Chat-hf" "29607"

# VSP - Visual spatial planning
bash /home/aiscuser/lmms-eval/g2u/emu3.sh "5" "vsp" "./logs/emu3/vsp" "BAAI/Emu3-Chat-hf" "29608"
bash /home/aiscuser/lmms-eval/g2u/emu3.sh "0" "babyvision" "./logs/emu3/babyvision" "BAAI/Emu3-Chat-hf" "29706" ""
bash /home/aiscuser/lmms-eval/g2u/emu3.sh "6" "phyx_simple" "./logs/emu3/phyx_simple" "BAAI/Emu3-Chat-hf" "29707" ""
bash /home/aiscuser/lmms-eval/g2u/emu3.sh "7" "geometry3k" "./logs/emu3/geometry3k" "BAAI/Emu3-Chat-hf" "29708" "" "100"
bash /home/aiscuser/lmms-eval/g2u/emu3.sh "3" "auxsolidmath_easy" "./logs/emu3/auxsolidmath_easy" "BAAI/Emu3-Chat-hf" "29709" "" "100" 



echo ""
echo "======================================"
echo "All Emu3-Chat evaluations completed!"
echo "Results saved to: ./logs/emu3/"
echo "======================================"
