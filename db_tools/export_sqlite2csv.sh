DB_PATH="${1:-results.db}"
sqlite3 -header -csv ${DB_PATH}.db "select * from torchmlir_benchmark;" > results.csv
