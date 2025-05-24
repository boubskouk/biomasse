import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import Optional, Tuple

class AnalyseurEnergies:
    """
    Classe pour analyser les performances énergétiques à partir de données de combustion
    """
    
    def __init__(self):
        self.df = None
        self.df_original = None
        
    def charger_donnees(self, chemin_fichier: str, skip_rows: int = 4) -> pd.DataFrame:
        """
        Charge et nettoie les données depuis un fichier CSV
        
        Args:
            chemin_fichier: Chemin vers le fichier CSV
            skip_rows: Nombre de lignes à ignorer au début
        """
        print(f"📁 Chargement du fichier: {chemin_fichier}")
        
        if not os.path.exists(chemin_fichier):
            raise FileNotFoundError(f"Fichier introuvable: {chemin_fichier}")
        
        try:
            # Tentative de lecture avec différents séparateurs
            self.df = pd.read_csv(chemin_fichier, skiprows=skip_rows)
            
            # Si une seule colonne, essayer avec un autre séparateur
            if len(self.df.columns) == 1:
                self.df = pd.read_csv(chemin_fichier, skiprows=skip_rows, sep=';')
            
            # Nettoyer les noms de colonnes
            self.df.columns = self.df.columns.str.strip()
            
            print(f"✅ Données chargées: {self.df.shape[0]} lignes, {self.df.shape[1]} colonnes")
            print(f"📊 Colonnes disponibles: {list(self.df.columns)}")
            
            return self.df
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            raise
    
    def nettoyer_donnees(self) -> pd.DataFrame:
        """
        Nettoie les données en supprimant les valeurs manquantes et aberrantes
        """
        if self.df is None:
            raise ValueError("Aucune donnée chargée. Utilisez charger_donnees() d'abord.")
        
        print("🧹 Nettoyage des données...")
        
        # Sauvegarder l'original
        self.df_original = self.df.copy()
        taille_initiale = len(self.df)
        
        # Supprimer les lignes avec des valeurs manquantes
        self.df = self.df.dropna()
        print(f"   - Lignes avec valeurs manquantes supprimées: {taille_initiale - len(self.df)}")
        
        # Supprimer les valeurs négatives pour CO (si la colonne existe)
        if 'CO' in self.df.columns:
            avant = len(self.df)
            self.df = self.df[self.df['CO'] >= 0]
            print(f"   - Lignes avec CO négatif supprimées: {avant - len(self.df)}")
        
        print(f"✅ Nettoyage terminé: {len(self.df)} lignes conservées")
        return self.df
    
    def calculer_efficacite_combustion(self) -> pd.DataFrame:
        """
        Calcule l'efficacité de combustion: CO / (CO + CO2)
        """
        colonnes_requises = ['CO', 'CO2']
        self._verifier_colonnes(colonnes_requises)
        
        print("⚡ Calcul de l'efficacité de combustion...")
        
        # Éviter la division par zéro
        denominateur = self.df['CO'] + self.df['CO2']
        self.df['efficacite_combustion'] = np.where(
            denominateur > 0,
            self.df['CO'] / denominateur,
            0
        )
        
        print("✅ Efficacité de combustion calculée")
        return self.df
    
    def calculer_performances_energetiques(self) -> pd.DataFrame:
        """
        Calcule les indicateurs de performance énergétique
        """
        colonnes_requises = ['GasTemp', 'FlueTemp', 'TC', 'Flow']
        self._verifier_colonnes(colonnes_requises)
        
        print("🔥 Calcul des performances énergétiques...")
        
        # Calculs des indicateurs
        self.df['rendement_thermique'] = (self.df['GasTemp'] - self.df['TC']) / self.df['GasTemp']
        self.df['pertes_conduit'] = (self.df['FlueTemp'] - self.df['TC']) / (self.df['GasTemp'] - self.df['TC'])
        self.df['efficacite_transfert'] = (self.df['GasTemp'] - self.df['FlueTemp']) / (self.df['GasTemp'] - self.df['TC'])
        
        # Calculs de puissance
        self.df['puissance_demandee'] = self.df['Flow'] * (self.df['GasTemp'] - self.df['TC'])
        self.df['puissance_utilisee'] = self.df['Flow'] * (self.df['GasTemp'] - self.df['FlueTemp'])
        
        # Efficacité globale (éviter division par zéro)
        self.df['efficacite_globale'] = np.where(
            self.df['puissance_demandee'] > 0,
            self.df['puissance_utilisee'] / self.df['puissance_demandee'],
            0
        )
        
        # Zone économique (seuil d'efficacité > 85%)
        self.df['zone_economique'] = self.df['efficacite_globale'] > 0.85
        
        print("✅ Performances énergétiques calculées")
        return self.df
    
    def afficher_statistiques(self):
        """
        Affiche les statistiques des performances
        """
        if self.df is None:
            raise ValueError("Aucune donnée disponible")
        
        print("\n📊 STATISTIQUES DES PERFORMANCES")
        print("=" * 50)
        
        # Statistiques de combustion si disponibles
        if 'efficacite_combustion' in self.df.columns:
            print("\n🔥 Efficacité de combustion:")
            print(self.df['efficacite_combustion'].describe())
        
        # Statistiques énergétiques si disponibles
        colonnes_perf = ['rendement_thermique', 'pertes_conduit', 'efficacite_transfert', 'efficacite_globale']
        colonnes_disponibles = [col for col in colonnes_perf if col in self.df.columns]
        
        if colonnes_disponibles:
            print(f"\n⚡ Performances énergétiques:")
            print(self.df[colonnes_disponibles].describe())
            
            # Pourcentage en zone économique
            if 'zone_economique' in self.df.columns:
                pct_economique = (self.df['zone_economique'].sum() / len(self.df)) * 100
                print(f"\n💰 Zone économique (>85%): {pct_economique:.1f}% des mesures")
    
    def visualiser_performances(self, colonnes_temps: Optional[str] = None):
        """
        Génère les graphiques de performance
        
        Args:
            colonnes_temps: Nom de la colonne temporelle (par défaut cherche 'seconds', 'time', etc.)
        """
        if self.df is None:
            raise ValueError("Aucune donnée disponible")
        
        # Trouver la colonne temporelle
        if colonnes_temps is None:
            colonnes_temps_possibles = ['seconds', 'time', 'temps', 'timestamp']
            for col in colonnes_temps_possibles:
                if col in self.df.columns:
                    colonnes_temps = col
                    break
        
        print(f"📈 Génération des graphiques...")
        
        # Configuration du style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # Graphique 1: Efficacité de combustion
        if 'efficacite_combustion' in self.df.columns and colonnes_temps:
            plt.figure(figsize=(12, 4))
            plt.plot(self.df[colonnes_temps], self.df['efficacite_combustion'], 
                    color='red', linewidth=1.5, alpha=0.8)
            plt.title('Évolution de l\'efficacité de combustion')
            plt.xlabel(f'Temps ({colonnes_temps})')
            plt.ylabel('CO / (CO + CO2)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        # Graphique 2: Efficacité globale
        if 'efficacite_globale' in self.df.columns and colonnes_temps:
            plt.figure(figsize=(12, 4))
            plt.plot(self.df[colonnes_temps], self.df['efficacite_globale'], 
                    color='blue', linewidth=1.5, alpha=0.8, label='Efficacité Globale')
            plt.axhline(0.85, color='green', linestyle='--', alpha=0.7, 
                       label='Seuil zone économique (85%)')
            plt.title('Efficacité Globale dans le Temps')
            plt.xlabel(f'Temps ({colonnes_temps})')
            plt.ylabel('Efficacité Globale')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1.2)
            plt.tight_layout()
            plt.show()
        
        # Graphique 3: Distribution des performances
        colonnes_perf = ['rendement_thermique', 'efficacite_transfert', 'efficacite_globale']
        colonnes_disponibles = [col for col in colonnes_perf if col in self.df.columns]
        
        if len(colonnes_disponibles) >= 2:
            fig, axes = plt.subplots(1, len(colonnes_disponibles), figsize=(4*len(colonnes_disponibles), 4))
            if len(colonnes_disponibles) == 1:
                axes = [axes]
            
            for i, col in enumerate(colonnes_disponibles):
                axes[i].hist(self.df[col], bins=30, alpha=0.7, color=f'C{i}')
                axes[i].set_title(f'Distribution - {col.replace("_", " ").title()}')
                axes[i].set_xlabel('Valeur')
                axes[i].set_ylabel('Fréquence')
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    def sauvegarder_resultats(self, chemin_sortie: str = None):
        """
        Sauvegarde les résultats dans un fichier CSV
        """
        if self.df is None:
            raise ValueError("Aucune donnée à sauvegarder")
        
        if chemin_sortie is None:
            chemin_sortie = "resultats_analyse_energetique.csv"
        
        self.df.to_csv(chemin_sortie, index=False)
        print(f"💾 Résultats sauvegardés: {chemin_sortie}")
        return chemin_sortie
    
    def _verifier_colonnes(self, colonnes_requises: list):
        """Vérifie que les colonnes requises sont présentes"""
        if self.df is None:
            raise ValueError("Aucune donnée chargée")
        
        colonnes_manquantes = [col for col in colonnes_requises if col not in self.df.columns]
        if colonnes_manquantes:
            print(f"❌ Colonnes manquantes: {colonnes_manquantes}")
            print(f"📋 Colonnes disponibles: {list(self.df.columns)}")
            raise ValueError(f"Colonnes manquantes: {colonnes_manquantes}")


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

def exemple_utilisation():
    """
    Exemple d'utilisation de l'analyseur énergétique
    """
    # Initialiser l'analyseur
    analyseur = AnalyseurEnergies()
    
    # Chemin du fichier (à adapter selon votre environnement)
    chemin_fichier = "/content/votre_fichier.csv"  # Remplacez par votre chemin
    
    try:
        # 1. Charger et nettoyer les données
        analyseur.charger_donnees(chemin_fichier, skip_rows=4)
        analyseur.nettoyer_donnees()
        
        # 2. Calculer les performances (selon les colonnes disponibles)
        try:
            analyseur.calculer_efficacite_combustion()
        except ValueError as e:
            print(f"⚠️  Efficacité de combustion non calculée: {e}")
        
        try:
            analyseur.calculer_performances_energetiques()
        except ValueError as e:
            print(f"⚠️  Performances énergétiques non calculées: {e}")
        
        # 3. Afficher les statistiques
        analyseur.afficher_statistiques()
        
        # 4. Générer les graphiques
        analyseur.visualiser_performances()
        
        # 5. Sauvegarder les résultats
        analyseur.sauvegarder_resultats()
        
    except Exception as e:
        print(f"❌ Erreur: {e}")


# Pour utiliser ce code dans Colab, décommentez la ligne suivante:
# exemple_utilisation()