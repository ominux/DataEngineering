sudo pip install -r requirements.txt
sudo pip install . --upgrade
# sudo pip3 install -r requirements.txt
# sudo pip3 install . --upgrade

#-------------------------------
# Run all tests
# Requires each directory to path to any test files to have __init__.py to discover it
# But this will mess up your original __init__.py to libraries.
# Thus, put test files within same directory as source files
# Also requires all test file names to start with test_*.py or test*.py
# nosetests will find files from current directory inwards
# therefore, run it within a specific directory if you only wanna test files in that directory
nosetests -vv --nocapture 
#-------------------------------
# Remove temporary files
find . -name '*.pyc' -delete
