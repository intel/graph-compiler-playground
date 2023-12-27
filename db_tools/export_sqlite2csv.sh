DB_PATH="${1:-results.db}"
sqlite3 -header -csv results.db "select * from torchmlir_benchmark;" > results.csv
                  name: ${{ steps.set_up_vars.outputs.conda_env }}.db
