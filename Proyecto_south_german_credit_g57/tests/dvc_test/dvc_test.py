import csv

ONLY_COUNT = False
ORIGINAL_COUNT = 1000
DB_FILE_PATH = "..\\..\\data\\raw\\german_credit_original.csv"
SYNTHETIC_RECORD = [2, 12, 2, 3, 6468, 5, 1, 2, 3, 1, 1, 4, 52, 3, 2, 1, 4, 2, 2, 2, 0]

"""
    Cuenta el número de registros en la base de datos original
    <German South Credit>.
    No retorna ningún valor.

    Parameters
    ----------
    db_file_path: string
        La localización exacta del archivo <German South Credit>
        al cual se le agregará un registro al final.
    has_header: boolean
        Si el archivo tiene o no encabezados.
        Por defecto sí tiene.

    Returns
    -------
    void
        No retorna valores.
"""


def count_records_on_db(db_file_path, has_header=True):
    with open(db_file_path, 'r', newline='') as south_german_credit_db:
        reader = csv.reader(south_german_credit_db)
        record_count = sum(1 for row in reader)
        if has_header and record_count > 0:
            return record_count - 1
        return record_count


"""
    Incrementa en una unidad el número de registros en la base de datos original.
    No retorna ningún valor.

    Parameters
    ----------
    db_file_path: string
        La localización exacta del archivo <German South Credit>
        al cual se le agregará un registro al final.
    record_data:
        El registro a ser añadido al final del archivo.
    Returns
    -------
    void
        No retorna valores.
"""


def insert_record_on_db(db_file_path, synthetic_record_one):
    try:
        with open(db_file_path, 'a', newline='') as south_german_credit_db:
            csv_writer = csv.writer(south_german_credit_db, lineterminator='\n')
            csv_writer.writerow(synthetic_record_one)
            print(f"Se añadió un registro al final de la base de datos.")
    except FileNotFoundError:
        print(f"Error al escribir en el Archivo: No encontrado.")
    except Exception as e:
        print(f"Error al escribir en el Archivo: {e}")


if __name__ == "__main__":
    no_records = count_records_on_db(DB_FILE_PATH)
    print("No. de registros en el archivo actualmente: {}".format(no_records))

    if not ONLY_COUNT:
        insert_record_on_db(DB_FILE_PATH, SYNTHETIC_RECORD)

    if ORIGINAL_COUNT == no_records:
        print("El archivo está según el número de registros originales.")
    else:
        print("El archivo ha cambiado el número de registros originales.")
