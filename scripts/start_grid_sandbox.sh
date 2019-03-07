sh ./scripts/delete_all_dbs.sh

for i in {5000..5010}
do
	screen -d -m -S flask_$i sh ./scripts/start_grid_node.sh localhost $i
done

