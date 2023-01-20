#%%
import random
import math
import matplotlib.pyplot as plt
import heapq
import time
import numpy as np
from typing import List
from typing import Union
from collections import deque

def generate_random_graph(n, m):
    """
    Cette fonction prend en entrée deux entiers n et m, et génère un graphe aléatoire contenant n sommets et m arêtes. 
    Elle retourne le graphe sous forme de dictionnaire, où chaque clé représente un sommet et la valeur associée est un dictionnaire des arêtes sortantes du sommet avec leur poids.
    """
    graph = {}
    # Initialisation d'un dictionnaire vide pour stocker les sommets
    for i in range(n):
        graph[i] = {}
    # Boucle pour générer les arêtes
    for i in range(m):
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        # Boucle pour éviter les arêtes bouclantes et les arêtes multiples
        while u == v or v in graph[u]:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
        w = random.uniform(0, 1)
        graph[u][v] = w
        graph[v][u] = w
    return graph

def display_graph(graph):
    """
    Cette fonction prend en entrée un graphe sous forme de dictionnaire et affiche ce graphe sous forme de graphique en utilisant la bibliothèque matplotlib.
    """
    # Initialisation d'une liste vide pour stocker les arêtes
    edges = []
    # Boucle pour parcourir les sommets du graphe
    for u in graph:
        for v, weight in graph[u].items():
            if (u, v) not in edges and (v, u) not in edges:
                edges.append((u, v))
    # Initialisation de la figure
    plt.figure(figsize=(8, 8))
    plt.grid(True)
    # Generation de coordonnées aléatoires pour les sommets
    n = len(graph)
    coordinates = np.random.rand(n,2)
    # Boucle pour tracer les sommets
    for vertex in range(n):
        plt.plot(coordinates[vertex][0], coordinates[vertex][1], "o")
    # Boucle pour tracer les arêtes
    for u, v in edges:
        plt.plot([coordinates[u][0], coordinates[v][0]], [coordinates[u][1], coordinates[v][1]], "k-")
    # Affichage de la figure
    plt.show()

def complete_graph(points):
    """
    Cette fonction prend en entrée une liste de points et retourne un graphe complet,
    où chaque point est relié à tous les autres points de la liste avec un poids
    calculé à l'aide de la distance euclidienne entre les points.
    """
    n = len(points)
    graph = {}
    for i in range(n):
        graph[i] = {}
        for j in range(n):
            if i != j:
                graph[i][j] = euclidian(points[i], points[j])
    return graph

def euclidian(p1, p2):
    # Calcule la distance euclidienne entre deux points
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

# Fonction qui calcule la distance d'un cycle
def cycle_distance(graph: dict, cycle: List) -> Union[int, None]:
    total = 0
    n = len(cycle)
    for i in range(n):
        u, v = cycle[i], cycle[(i+1) % n]
        if u in graph and v in graph[u]:
            total += graph[u][v]
    return total

# Algorithme du plus proche voisin
def Ppvoisin(graph, start):
    # Cycle en cours de construction
    cycle = [start]
    # Sommet actuel
    current = start
    # Ensemble des sommets déjà visités
    visited = set()
    visited.add(start)
    # Répétition tant qu'il reste des sommets non visités
    while len(visited) < len(graph):
        # Sommet le plus proche parmi les sommets non visités
        nearest = None
        nearest_distance = float("inf")
        for neighbor in graph[current]:
            if neighbor not in visited and graph[current][neighbor] < nearest_distance:
                nearest = neighbor
                nearest_distance = graph[current][neighbor]
        # Si aucun sommet n'a été trouvé, cela signifie qu'il n'y a pas de cycle valide dans le graphe
        if nearest == None:
            # On ajoute un sommet aléatoire non visité au cycle
            unvisited = set(graph.keys()) - visited
            nearest = unvisited.pop()
            cycle.append(nearest)
            visited.add(nearest)
            current = nearest
            continue
        # Ajout du sommet le plus proche au cycle
        cycle.append(nearest)
        visited.add(nearest)
        current = nearest
    # Ajout de l'arête qui ferme le cycle
    cycle.append(start)
    return cycle, cycle_distance(graph, cycle)

def OptimisePpvoisin(L, graph):
    improved = True
    while improved:
        improved = False
        for i in range(len(L)):
            for j in range(i+2, len(L)):
                # On vérifie si les arêtes se croisent
                if (L[i], L[(i+1) % len(L)]) == (L[j], L[(j+1) % len(L)]):
                    continue
                # On calcule les distances avant et après décroisement
                dist_before = graph[L[i]][L[(i+1)%len(L)]] + graph[L[j]][L[(j+1)%len(L)]]
                dist_after = graph[L[i]][L[j]] + graph[L[(i+1)%len(L)]][L[(j+1)%len(L)]]
                if dist_after < dist_before:
                    improved = True
                    L[i+1:j+1] = reversed(L[i+1:j+1])
    return L, cycle_distance(graph, L)


# Algorithme de l'arête de poids minimum
def Apminimum(graph, start):
    # Cycle en cours de construction
    cycle = [start]
    # Sommet actuel
    current = start
    # Ensemble des sommets déjà visités
    visited = set()
    visited.add(start)
    # Répétition tant qu'il reste des sommets non visités
    while len(visited) < len(graph):
        # Arête de poids minimum parmi les arêtes qui ne referment pas prématurément le cycle
        min_edge = None
        min_distance = float("inf")
        for neighbor in graph[current]:
            if neighbor not in visited and graph[current][neighbor] < min_distance:
                min_edge = (current, neighbor)
                min_distance = graph[current][neighbor]
        # Si aucun sommet n'a été trouvé, cela signifie qu'il n'y a pas de cycle valide dans le graphe
        if min_edge == None:
            return None
        # Ajout du sommet lié par l'arête de poids minimum au cycle
        cycle.append(min_edge[1])
        visited.add(min_edge[1])
        current = min_edge[1]
    # Ajout de l'arête qui ferme le cycle
    cycle.append(start)
    return cycle, cycle_distance(graph, cycle)

# Fonction de l'algorithme de Prim
def prim(graph, start):
    # Arbre couvrant de poids minimum en cours de construction
    tree = {start: {}}
    # Ensemble des sommets déjà ajoutés à l'arbre couvrant de poids minimum
    visited = set()
    visited.add(start)
    # Répétition tant qu'il reste des sommets non visités
    while len(visited) < len(graph):
        # Arête de poids minimum parmi les arêtes reliant un sommet de l'arbre couvrant de poids minimum
        # à un sommet non encore ajouté à l'arbre couvrant de poids minimum
        min_edge = None
        min_distance = float("inf")
        for u in tree:
            for v in graph[u]:
                if v not in visited and graph[u][v] < min_distance:
                    min_edge = (u, v)
                    min_distance = graph[u][v]
        # Ajout du sommet lié par l'arête de poids minimum à l'arbre couvrant de poids minimum
        tree[min_edge[1]] = {}
        tree[min_edge[0]][min_edge[1]] = min_distance
        visited.add(min_edge[1])
    return tree

# Fonction qui calcule un cycle hamiltonien du graphe qui visite les sommets de l'arbre couvrant de poids minimum
# construit par l'algorithme de Prim, dans l'ordre préfixe
def Pvcprim(graph, start):
    # Construction de l'arbre couvrant de poids minimum
    tree = prim(graph, start)
    # Vérification que l'arbre est connecté
    if not is_connected(tree):
        return None
    # Construction du cycle hamiltonien en parcourant l'arbre couvrant de poids minimum dans l'ordre préfixe
    cycle = [start]
    stack = [start]
    while stack:
        current = stack[-1]
        for neighbor in tree[current]:
            if neighbor not in cycle:
                cycle.append(neighbor)
                stack.append(neighbor)
                break
        else:
            stack.pop()
    # Ajout de l'arête qui ferme le cycle
    cycle.append(start)
    return cycle, cycle_distance(graph, cycle)

def is_connected(graph):
    """
    Cette fonction prend en entrée un graphe sous forme de dictionnaire, et vérifie si ce graphe est connecté ou non.
    Elle utilise une recherche en largeur pour parcourir tous les sommets du graphe.
    Elle retourne True si le graphe est connecté, False sinon.
    """
    # Vérifie si le graphe est vide
    if not graph:
        return False
    # Sommet de départ
    start = list(graph.keys())[0]
    # Initialisation de la file d'attente pour la recherche en largeur
    queue = deque([start])
    # Ensemble des sommets visités
    visited = set()
    visited.add(start)
    # Boucle de recherche en largeur
    while queue:
        current = queue.popleft()
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    # Si tous les sommets ont été visités, cela signifie que le graphe est connecté
    return len(visited) == len(graph)

# Algorithme de résolution du PVC par évaluation et séparation progressive
def Esdemisomme(graph, start):
    # Fonction récursive qui parcourt l'arborescence des possibilités et qui choisit, à chaque étape,
    # le successeur dont la valeur de l'heuristique est la plus faible
    def search(graph, path, remaining):
        # Calcul de la demi-somme des poids des arêtes restantes
        half_sum = sum(graph[path[-1]][v] for v in remaining) / 2
        if not remaining:
            # Si il n'y a plus de successeurs, le cycle est complet
            return path, half_sum
        # Si la valeur de l'heuristique est supérieure à la meilleure solution trouvée, on coupe la branche
        if half_sum > best_distance:
            return None
        # Liste des successeurs qui mènent à une solution intéressante
        good_successors = []
        for v in remaining:
            successor = search(graph, path + [v], remaining - {v})
            if successor:
                good_successors.append(successor)
        # Tri des successeurs par ordre croissant de la valeur de l'heuristique
        good_successors.sort(key=lambda x: x[1])
        # Renvoi du premier successeur (celui qui a la valeur de l'heuristique la plus faible)
        return good_successors[0]

    # Initialisation de la meilleure solution trouvée
    best_solution = None
    best_distance = float("inf")
    # Parcours de l'arborescence des possibilités avec la fonction search
    solution = search(graph, [start], set(graph) - {start})
    if solution:
        best_solution = solution[0]
        best_distance = solution[1]
    # Renvoi du cycle optimal et de son poids
    return best_solution, cycle_distance(graph, best_solution)

# Programme principal
if __name__ == "__main__":
    # Génération aléatoire de n points dans l'intervalle [0, 1]
    n = int(input("Entrez la valeur de n :"))
    points = []
    for i in range(n):
        points.append((random.random(), random.random()))
    # Création du graphe complet à partir de ces points
    graph = complete_graph(points)
    # Affichage du graphe
    display_graph(graph)
    # Calcul et affichage des cycles et de leurs poids obtenus par chacun des algorithmes
    cycle_ppvoisin, distance_ppvoisin = Ppvoisin(graph, 0)
    print("Algorithme du plus proche voisin :", cycle_ppvoisin, distance_ppvoisin)
    #print("Algoritme du plus proche voisin optimisé :", OptimisePpvoisin(cycle_ppvoisin, graph))
    print("Algorithme de l'arête de poids minimum :", Apminimum(graph, 0))
    print("Algorithme de Prim :", Pvcprim(graph, 0))
    print("Algorithme par évaluation et séparation progressive :", Esdemisomme(graph, 0))

    # Etude de la qualité des résultats fournis par les quatre algorithmes
    # en calculant, sur 100 essais, la longueur moyenne des cycles obtenus par chacune des méthodes implantées
    nb_tests = 100
    distances_ppvoisin = []
    distances_optppvoisin = []
    distances_apm = []
    distances_pvcprim = []
    distances_pvcbb = []
    for i in range(nb_tests):
     # Génération aléatoire de n points dans l'intervalle [0, 1]
     points = []
     for i in range(n):
        points.append((random.random(), random.random()))
        # Création du graphe complet à partir de ces points
        graph = complete_graph(points)
        # Calcul des cycles et de leurs poids obtenus par chacun des algorithmes
        cycle_ppvoisin, distance_ppvoisin = Ppvoisin(graph, 0)
        #cycl_optppvoisin, distance_optppvoisin = OptimisePpvoisin(cycle_ppvoisin, graph)
        cycle_apm, distance_apm = Apminimum(graph, 0)
        cycle_pvcprim, distance_pvcprim = Pvcprim(graph, 0)
        cycle_demisomme, distance_pvcbb = Esdemisomme(graph, 0)
        # Enregistrement des distances dans des listes
        distances_ppvoisin.append(distance_ppvoisin)
        #distances_optppvoisin.append(distance_optppvoisin)
        distances_apm.append(distance_apm)
        distances_pvcprim.append(distance_pvcprim)
        distances_pvcbb.append(distance_pvcbb)
        # Calcul des moyennes des distances obtenues par chaque algorithme
        mean_ppvoisin = sum(distances_ppvoisin) / nb_tests
        #mean_optppvoisin = sum(distance_optppvoisin) / nb_tests
        mean_apm = sum(distances_apm) / nb_tests
        mean_pvcprim = sum(distances_pvcprim) / nb_tests
        mean_pvcbb = sum(distances_pvcbb) / nb_tests
        
    # Affichage des moyennes des distances pour chaque algorithmes
    print("Moyenne des distances pour l'algorithme du plus proche voisin :", mean_ppvoisin)
    print("Moyenne des distances pour l'algorithme de l'arête de poids minimum :", mean_apm)
    print("Moyenne des distances pour l'algorithme de prim :", mean_pvcprim)
    print("Moyenne des distances pour l'algorithme par évaluation et séparation progressive :", mean_pvcbb)
    # Calcul et affichage des pourcentages gagnés par chaque méthode par rapport aux autres
    print("Pourcentage de gain de l'algorithme du plus proche voisin par rapport à l'algorithme de l'arête de poids minimum :", (mean_apm - mean_ppvoisin) / mean_apm * 100, "%")
    #print("Pourcentage de gain de l'algorithme du plus proche voisin par rapport a l'algorithme du plus proche voisin optimisé :", (mean_optppvoisin - mean_ppvoisin) / mean_optppvoisin * 100, "%")
    print("Pourcentage de gain de l'algorithme du plus proche voisin par rapport à l'algorithme de Prim :", (mean_pvcprim - mean_ppvoisin) / mean_pvcprim * 100, "%")
    print("Pourcentage de gain de l'algorithme du plus proche voisin par rapport à l'algorithme par évaluation et séparation progressive :", (mean_pvcbb - mean_ppvoisin) / mean_pvcbb * 100, "%")
   # print("Pourcentage de gain de l'algorithme du plus proche voisin optimisé par rapport a l'algorithme de prim", (mean_pvcprim - mean_optppvoisin) / mean_pvcprim * 100, "%")
   # print("Pourcentage de gain de l'algorithme du plus proche voisin optimisé par rapport a l'algorithme par évaluation et séparation progressive", (mean_pvcbb - mean_optppvoisin) / mean_pvcbb * 100, "%")
    print("Pourcentage de gain de l'algorithme de l'arête de poids minimum par rapport à l'algorithme de Prim :", (mean_pvcprim - mean_apm) / mean_pvcprim * 100, "%")
    print("Pourcentage de gain de l'algorithme de l'arête de poids minimum par rapport à l'algorithme par évaluation et séparation progressive :", (mean_pvcbb - mean_apm) / mean_pvcbb * 100, "%")
   # print("Pourcentage de gain de l'algorithme de l'arête de poids minimum par rapport à l'algorithme du plus proche voisin optimisé", (mean_optppvoisin - mean_apm) / mean_optppvoisin * 100, "%")
    print("Pourcentage de gain de l'algorithme de Prim par rapport à l'algorithme par évaluation et séparation progressive :", (mean_pvcbb - mean_pvcprim) / mean_pvcbb * 100, "%")

    # Etude sur le nombre maximum de sommets pour lequel le temps d'exécution de votre programme reste raisonnable
    nb_points = range(2, 10)
    times_ppvoisin = []
    times_optppvoisin = []
    times_apm = []
    times_pvcprim = []
    times_pvcbb = []
    for n in nb_points:
        # Génération aléatoire de n points dans l'intervalle [0, 1]
        points = []
        for i in range(n):
            points.append((random.random(), random.random()))
        # Création du graphe complet à partir de ces points
        graph = complete_graph(points)
        # Calcul des temps d'exécution de chaque algorithme
        start_time = time.time()
        Ppvoisin(graph, 0)
        end_time = time.time()
        times_ppvoisin.append(end_time - start_time)
        start_time = time.time()
        #OptimisePpvoisin(graph, Ppvoisin(graph, 0))
        #end_time = time.time()
        #times_optppvoisin.append(end_time - start_time)
        #start_time = time.time()
        Apminimum(graph, 0)
        end_time = time.time()
        times_apm.append(end_time - start_time)
        start_time = time.time()
        Pvcprim(graph, 0)
        end_time = time.time()
        times_pvcprim.append(end_time - start_time)
        start_time = time.time()
        Esdemisomme(graph, 0)
        end_time = time.time()
        times_pvcbb.append(end_time - start_time)

    # Représentation visuelle de l'étude statistique
    plt.plot(nb_points, times_ppvoisin, label="Algorithme du plus proche voisin")
    #plt.plot(nb_points, times_optppvoisin, label="Algorithme du plus proche voisin optimisé")
    plt.plot(nb_points, times_apm, label="Algorithme de l'arête de poids minimum")
    plt.plot(nb_points, times_pvcprim, label="Algorithme de Prim")
    plt.plot(nb_points, times_pvcbb, label="Algorithme par évaluation et séparation progressive")
    plt.xlabel("Nombre de points")
    plt.ylabel("Temps d'exécution (en secondes)")
    plt.legend()
    plt.show()





# %%
