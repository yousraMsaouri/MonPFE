import os
import sys
import pandas as pd
import psutil

# 1. Chargement de la DLL Microsoft ADOMD
dll_path = r"C:\Program Files\Microsoft.NET\ADOMD.NET\160\Microsoft.AnalysisServices.AdomdClient.dll"
if os.path.exists(dll_path):
  sys.path.append(os.path.dirname(dll_path))
  os.environ["PATH"] += os.pathsep + os.path.dirname(dll_path)

import clr

clr.AddReference(dll_path)

# Import des classes .NET natives de Microsoft
from Microsoft.AnalysisServices.AdomdClient import AdomdConnection
from pyadomd import Pyadomd


# 2. Détection du port local Power BI
def obtenir_port_powerbi():
  for proc in psutil.process_iter(["name"]):
    try:
      if proc.info["name"] == "msmdsrv.exe":
        for conn in proc.net_connections():
          if conn.status == "LISTEN":
            return conn.laddr.port
    except (psutil.NoSuchProcess, psutil.AccessDenied):
      continue
  return None


# 3. Extraction dynamique du nom du catalogue via .NET
def obtenir_nom_catalogue(port):
  conn_str_base = f"Data Source=localhost:{port};"
  try:
    net_conn = AdomdConnection(conn_str_base)
    net_conn.Open()

    # Récupération de la liste des catalogues hébergés sur l'instance locale
    schema_db = net_conn.GetSchemaDataSet("DBSCHEMA_CATALOGS", None)
    net_conn.Close()

    if schema_db and schema_db.Tables.Count > 0:
      # Extraction du premier nom de catalogue trouvé
      catalog_name = schema_db.Tables[0].Rows[0]["CATALOG_NAME"]
      return str(catalog_name)
  except Exception as err:
    print(
        f"⚠️ Impossible d'extraire le catalogue via .NET : {err}"
    )  # Ne bloque pas le programme
  return None


# 4. Exécution
port = obtenir_port_powerbi()

if not port:
  print("❌ Erreur : Veillez à ce que le fichier Power BI Desktop soit ouvert.")
else:
  print(f"✅ Port local identifié : {port}")

  catalog = obtenir_nom_catalogue(port)

  if catalog:
    print(f"✅ Catalogue (GUID) extrait avec succès : {catalog}")
    conn_str = f"Provider=MSOLAP;Data Source=localhost:{port};Initial Catalog={catalog};"

    # Requête DAX pour vérifier la connexion aux tables
    dax_query = (
        'EVALUATE TOPN(10, SELECTCOLUMNS(INFO.TABLES(), "TableName", [Name]))'
    )

    try:
      with Pyadomd(conn_str) as conn:
        with conn.cursor() as cursor:
          cursor.execute(dax_query)
          data = cursor.fetchall()
          cols = [col[0] for col in cursor.description]
          df_tables = pd.DataFrame(data, columns=cols)

      print("\n🎉 CONNEXION RÉUSSIE ! Liste des tables disponibles :")
      print(df_tables)

    except Exception as e:
      print(f"⚠️ Erreur lors de l'exécution DAX : {e}")
  else:
    print("❌ Impossible de déterminer le catalogue local de Power BI.")