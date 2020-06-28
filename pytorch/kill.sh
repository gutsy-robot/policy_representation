kill -9 `ps aux | grep 'python pytorch/train_agent_classifier.py' | awk '{print $2}'`
ps aux | grep 'python pytorch/train_agent_classifier.py'