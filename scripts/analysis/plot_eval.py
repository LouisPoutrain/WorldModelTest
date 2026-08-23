import csv
import os
import sys

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def analyze_eval_logs(log_path="logs/eval_metrics_v2.csv"):
    if not os.path.exists(log_path):
        print(f"❌ Fichier {log_path} introuvable. Lancez eval_behavior.py d'abord.")
        return
        
    data = []
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
            
    if not data:
        print("Fichier CSV vide.")
        return
        
    print(f"✅ Chargement de {len(data)} pas d'évaluation.")
    print("="*60)
    print("📊 ANALYSE DES COLLISIONS (WALL HITS)")
    print("="*60)
    
    tests = {}
    for row in data:
        t = row['Test_Name']
        if t not in tests:
            tests[t] = {'total_steps': 0, 'wall_hits': 0}
        tests[t]['total_steps'] += 1
        if row['Wall_Hit'] == 'True':
            tests[t]['wall_hits'] += 1
            
    for t, stats in tests.items():
        hits = stats['wall_hits']
        total = stats['total_steps']
        pct = (hits / total) * 100 if total > 0 else 0
        print(f"Test {t:10s} : {hits:4d} collisions sur {total:4d} pas ({pct:.1f}%)")
        
    print("" + "="*60)
    print("🔍 DIAGNOSTIC PROFOND : ÉPISODE 0 DU U-TRAP")
    print("="*60)
    
    u_trap_ep0 = [r for r in data if r['Test_Name'] == 'U-Trap' and r['Episode_ID'] == '0']
    
    if u_trap_ep0:
        print(f"  {'Step':<5} | {'Position':<10} | {'Action':<8} | {'Mur ?':<6} | {'Dist Latente (Réelle)':<22} | {'Coût Prédit (CEM)':<18}")
        print("-" * 80)
        for row in u_trap_ep0[:25]: # Afficher les 25 premiers pas
            mur = "💥 OUI" if row['Wall_Hit'] == 'True' else "  Non"
            print(f"  {row['Step']:<5} | {row['Agent_Pos']:<10} | {row['Action']:<8} | {mur:<6} | {row['Latent_Dist_to_Goal']:<22} | {row['CEM_Predicted_Cost']:<18}")
            
        print("💡 INTERPRÉTATION :")
        print("- Si l'Agent reste bloqué sur la même position et que le 'Mur ?' est OUI en boucle :")
        print("  C'est parce que le Coût Prédit CEM est minimal en fonçant dans le mur.")
        print("- Si le CEM prévoit un coût très faible (ex: 2.0) mais que la Distance Réelle ne baisse pas :")
        print("  Cela prouve que le CEM ne trouve pas de trajectoire pour contourner (qui nécessiterait une distance temporairement plus élevée).")
    else:
        print("Aucune donnée U-Trap Ep 0 trouvée.")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "logs/eval_metrics_v2.csv"
    analyze_eval_logs(path)
