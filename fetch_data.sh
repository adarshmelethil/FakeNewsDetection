
data_folder="test_data"
mkdir -p $data_folder

d1="$data_folder/RTE1_dev1_3ways.xml"
if ! [[ -f "$FILE" ]]; then
  curl https://nlp.stanford.edu/projects/contradiction/RTE1_dev1_3ways.xml >> $d1
fi
