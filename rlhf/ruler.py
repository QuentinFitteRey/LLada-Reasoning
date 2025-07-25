import asyncio
from art import Trajectory, TrajectoryGroup
from art.rewards import ruler_score_group
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
import os
import math
# os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OPENAI_API_KEY"] = "ollama"  # required dummy

def scalling(scores):
    s_min, s_max = min(scores) + 1e-6, max(scores)
    span = s_max - s_min
    reward = [(s - s_min) / span for s in scores]
    return reward

def id(x):
    return x

def softmax(xs):
    # numerically stable softmax
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    S   = sum(exps)
    return [e / S for e in exps]


async def ruler_reward(prompts, answers, normalize_fn=scalling, judge_model="ollama/qwen3:30b-a3b-fp16"):
    # Setup the shared system/user prompt
    rewards = []
    for i in range(len(prompts)):
        print(f"Processing prompt {i + 1}/{len(prompts)}")
        messages = [
            {"role":"system", "content":"Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\n\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:"},
            {"role":"user", "content":prompts[i]},
        ]

        trajectories = []

        # Add the original answer as the first trajectory
        for answer in answers[i*len(answers)//len(prompts):(i+1)*len(answers)//len(prompts)]:
            trajectories.append(
                Trajectory(messages_and_choices=[
                    *messages,
                    Choice(finish_reason="stop", index=0, message=ChatCompletionMessage(
                        role="assistant",
                        content=answer
                    ))
                ], reward=0.0)
            )

        custom_rubric = """
    You are an impartial judge. Your task is to evaluate the quality of the AI-generated answer provided below based on two criteria: the correctness of the final solution and its strict adherence to the required response format.

    Provide a single floating-point score between 0.0 (lowest quality) and 1.0 (highest quality).

    EVALUATION CRITERIA

    1. Required Response Format:
    The response MUST be structured with the following two sections, using the exact delimiters:

    A "Thought" section, enclosed by <|begin_of_thought|> and <|end_of_thought|>, which details the step-by-step reasoning.

    A "Solution" section, enclosed by <|begin_of_solution|> and <|end_of_solution|>, which presents the final, concise answer.

    2. Correctness:
    The final answer within the <|begin_of_solution|> section must be factually accurate and logically sound.

    SCORING RUBRIC

    1.0: The solution is completely correct and the format is followed perfectly.

    0.5 - 0.9: The response is either correct with formatting errors OR follows the format but contains errors in the solution.

    0.0 - 0.4: The response is fundamentally incorrect and fails to adhere to the required format.

    Your output must contain the numerical score. You can provide explanations.
        """
        group = TrajectoryGroup(trajectories)
        # print("Here!")
        # print("Evaluating group with custom rubric...")
        try:
            judged = await ruler_score_group(group, judge_model, rubric=custom_rubric, extra_litellm_params={"api_base": "http://127.0.0.1:11435"})
            if judged: 
                scores = [traj.reward for traj in list(judged.trajectories)]
                reward = normalize_fn(scores)
                rewards += reward
                # print(reward)
                # sorted_trajectories = sorted(judged.trajectories, key=lambda t: t.reward, reverse=True)
                # for rank, traj in enumerate(sorted_trajectories, 1):
                #     messages = traj.messages()
                #     print(f"\n\n\nRank {rank}: Score {traj.reward:.3f}")
                #     print(f"  Response: {messages[-1]['content'][:250]}...")
            else:
                raise Exception("Judging failed, no trajectories returned.")
        except Exception as e:
            print(f"Error during judging: {e}")
            reward = [0.0 for _ in range(len(answers))]
            rewards += reward
    return rewards

def sync_ruler_reward(texts: list[str], prompt: str) -> list[float]:
    # run the async function to completion on the current event loop
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # If you’re already inside an asyncio loop (rare in a plain training script),
        # fall back to creating a new one:
        return asyncio.run(ruler_reward(prompt, texts))
    else:
        return loop.run_until_complete(ruler_reward(prompt, texts))


if __name__ == "__main__":
    prompt = "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition.You are given an undirected graph consisting of n vertices. A number is written on each vertex; the number on vertex i is a_i. Initially there are no edges in the graph.\n\nYou may add some edges to this graph, but you have to pay for them. The cost of adding an edge between vertices x and y is a_x + a_y coins. There are also m special offers, each of them is denoted by three numbers x, y and w, and means that you can add an edge connecting vertices x and y and pay w coins for it. You don't have to use special offers: if there is a pair of vertices x and y that has a special offer associated with it, you still may connect these two vertices paying a_x + a_y coins for it.\n\nWhat is the minimum number of coins you have to spend to make the graph connected? Recall that a graph is connected if it's possible to get from any vertex to any other vertex using only the edges belonging to this graph.\n\nInput\n\nThe first line contains two integers n and m (1 ≤ n ≤ 2 ⋅ 10^5, 0 ≤ m ≤ 2 ⋅ 10^5) — the number of vertices in the graph and the number of special offers, respectively.\n\nThe second line contains n integers a_1, a_2, ..., a_n (1 ≤ a_i ≤ 10^{12}) — the numbers written on the vertices.\n\nThen m lines follow, each containing three integers x, y and w (1 ≤ x, y ≤ n, 1 ≤ w ≤ 10^{12}, x ≠ y) denoting a special offer: you may add an edge connecting vertex x and vertex y, and this edge will cost w coins.\n\nOutput\n\nPrint one integer — the minimum number of coins you have to pay to make the graph connected.\n\nExamples\n\nInput\n\n\n3 2\n1 3 3\n2 3 5\n2 1 1\n\n\nOutput\n\n\n5\n\n\nInput\n\n\n4 0\n1 3 3 7\n\n\nOutput\n\n\n16\n\n\nInput\n\n\n5 4\n1 2 3 4 5\n1 2 8\n1 3 10\n1 4 7\n1 5 15\n\n\nOutput\n\n\n18\n\nNote\n\nIn the first example it is possible to connect 1 to 2 using special offer 2, and then 1 to 3 without using any offers.\n\nIn next two examples the optimal answer may be achieved without using special offers.\n"
    answer1 = """
<|begin_of_thought|>
The user wants me to solve a graph problem using a specific format.

1. Analyze the Problem:

The goal is to connect a graph of n vertices with minimum cost. This is a classic Minimum Spanning Tree (MST) problem.

We have n vertices, each with a value a_i.

There are two ways to add an edge between vertices x and y:

A "standard" edge with cost a_x + a_y.

A "special offer" edge with cost w, if one exists for the pair (x, y).

We can always choose the cheaper of the two options for a given pair.

2. Brainstorm Initial Ideas & Challenges:

An MST algorithm like Kruskal's or Prim's is the way to go. I'll use Kruskal's algorithm because it's generally easier to implement when the graph is defined by a list of edges.

The main challenge is the sheer number of potential edges. There are m special offers, but there are n * (n - 1) / 2 possible "standard" edges. Since n can be up to 2 * 10^5, generating all O(n^2) standard edges is too slow and will exceed the time limit.

3. Develop the Core Insight:

How can we avoid considering all O(n^2) standard edges?

Let's analyze the standard edge cost: a_x + a_y. This cost is minimized if we involve the vertex with the lowest a_i.

Let's find the vertex with the minimum a_i value. Call it v_min.

Any standard edge (u, v) with cost a_u + a_v can be compared to the path u -> v_min -> v. This path consists of two edges, (u, v_min) and (v, v_min), with costs a_u + a_{v_min} and a_v + a_{v_min} respectively.

Since a_{v_min} is the minimum possible a value, we know that a_u + a_{v_min} <= a_u + a_v and a_v + a_{v_min} <= a_v + a_u.

This means that in Kruskal's algorithm, the edges connecting to v_min would be considered no later than any other standard edge involving u or v. An MST built using only standard edges connecting to v_min will have a total weight less than or equal to an MST built using any arbitrary set of standard edges.

Therefore, we only need to consider the n-1 standard edges that connect v_min to every other vertex.

4. Formulate the Final Algorithm:

The complete set of candidate edges for our MST is the union of:

The m special offer edges (x, y) with their given cost w.

The n-1 standard edges (v_min, i) for all i != v_min, with cost a_{v_min} + a_i.

This gives us a total of m + n - 1 edges, which is manageable.

The overall algorithm is:
a. Read n, m, and the costs a_i.
b. Find the index min_idx of the vertex v_min with the minimum a_i.
c. Create a list of edges.
i. For each of the m special offers (x, y, w), add the edge (x, y, w) to the list.
ii. For each vertex i from 1 to n (if i != min_idx), add the edge (min_idx, i, a_{min_idx} + a_i) to the list.
d. Implement a Disjoint Set Union (DSU) data structure to keep track of connected components.
e. Sort the edge list by weight in ascending order.
f. Iterate through the sorted edges (u, v, weight) and apply Kruskal's algorithm:
i. If u and v are not already in the same component (checked with DSU find), add weight to the total cost and merge the components (with DSU union).
g. The final accumulated cost is the answer.

5. Refine and Verify:

The approach correctly identifies the problem as MST.

It correctly deals with the O(n^2) edge problem by finding a sufficient subset of n-1 standard edges.

The combination of these n-1 edges with the m special offer edges covers all necessary connections for a global minimum.

Kruskal's algorithm with DSU is the standard and correct way to find the MST cost from this edge list.

The logic is sound. I will proceed with implementation. I'll need to be careful with 1-based vs. 0-based indexing. The problem uses 1-based, so I'll adjust the input accordingly for 0-indexed Python lists.
<|end_of_thought|>
<|begin_of_solution|>
 import sys

def solve():
    \"\"\"
    This function solves the minimum cost connected graph problem
    by modeling it as a Minimum Spanning Tree (MST) problem.
    It uses Kruskal's algorithm with a Disjoint Set Union (DSU)
    data structure.

    The key insight is that we don't need to consider all O(n^2) possible
    standard edges (cost a_x + a_y). It's sufficient to only consider
    standard edges connecting every vertex to the single vertex with the
    minimum 'a' value (v_min). This is because any standard connection
    (u, v) can be achieved more cheaply or equally through v_min
    (i.e., path u -> v_min -> v).

    The algorithm proceeds as follows:
    1. Find the vertex v_min with the smallest associated cost a_i.
    2. Create a list of all potential edges for the MST:
       - The 'm' special offer edges.
       - The 'n-1' edges connecting v_min to all other vertices.
    3. Sort this combined list of edges by weight.
    4. Use a DSU data structure to build the MST by iterating through
       the sorted edges, adding an edge if it connects two previously
       unconnected components.
    5. Sum the weights of the edges added to the MST to get the
       minimum total cost.
    \"\"\"

    # Fast I/O
    input = sys.stdin.readline

    # DSU (Disjoint Set Union) data structure
    parent = []

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j
            return True
        return False

    # Read n and m
    n, m = map(int, input().split())

    # Read vertex costs
    a = list(map(int, input().split()))

    # Initialize DSU
    # Using 1-based indexing for vertices to match problem statement
    parent = list(range(n + 1))

    # Find the vertex with the minimum cost 'a'
    min_cost = float('inf')
    min_idx = -1
    for i in range(n):
        if a[i] < min_cost:
            min_cost = a[i]
            min_idx = i + 1

    # Create the list of edges
    edges = []

    # Add edges from the minimum cost vertex to all others
    for i in range(1, n + 1):
        if i == min_idx:
            continue
        cost = min_cost + a[i - 1]
        edges.append((min_idx, i, cost))

    # Add special offer edges
    for _ in range(m):
        x, y, w = map(int, input().split())
        edges.append((x, y, w))

    # Sort edges by weight
    edges.sort(key=lambda item: item[2])

    # Kruskal's algorithm
    total_cost = 0
    edges_count = 0
    for u, v, weight in edges:
        if union(u, v):
            total_cost += weight
            edges_count += 1
            # Optimization: stop when the graph is connected
            if edges_count == n - 1:
                break

    print(total_cost)

solve()
<|end_of_solution|>

"""
    answer2 = """
    <|begin_of_thought|>
    The problem asks for the minimum cost to make a graph connected. This is an MST problem. We can use Kruskal's algorithm.

    The main challenge is that there are O(n^2) possible edges if we consider connecting any pair of vertices with cost a_x + a_y. This is too slow.

    A better way is to notice that connecting to the cheapest vertex is always a good deal. Let's find the vertex with the minimum a_i, let's call it v_min. Any connection we need to make between two other vertices, say u and v, can be done through v_min. The path u-v_min-v is a candidate. This suggests we only need to consider standard edges from v_min to all other vertices.

    So, the set of edges to consider for the MST are:

    The m special offers.

    The n-1 edges connecting v_min to all other vertices.

    We can run Kruskal's algorithm on these m + n - 1 edges. This involves sorting the edges and using a Disjoint Set Union (DSU) structure to build the tree.
    <|end_of_thought|>
    <|begin_of_solution|>
    import sys

    def main():
        n, m = map(int, sys.stdin.readline().split())
        costs = list(map(int, sys.stdin.readline().split()))

        # Find the vertex with the minimum base cost
        min_a_val = min(costs)
        min_a_idx = costs.index(min_a_val) + 1

        edges = []

        # Add special offer edges
        for _ in range(m):
            u, v, w = map(int, sys.stdin.readline().split())
            edges.append((w, u, v))

        # Add edges from the cheapest vertex to all others
        for i in range(1, n + 1):
            if i != min_a_idx:
                cost = min_a_val + costs[i-1]
                edges.append((cost, min_a_idx, i))

        # Sort all candidate edges by weight
        edges.sort()

        parent = list(range(n + 1))

        def find_set(v):
            if v == parent[v]:
                return v
            parent[v] = find_set(parent[v])
            return parent[v]

        def union_sets(a, b):
            a = find_set(a)
            b = find_set(b)
            if a != b:
                parent[b] = a
                return True
            return False

        total_weight = 0
        num_edges = 0
        for weight, u, v in edges:
            if union_sets(u, v):
                total_weight += weight
                num_edges += 1
                if num_edges == n - 1:
                    break

        print(total_weight)

    main()
    <|end_of_solution|>

    
    """
    answer3 = """
    <|begin_of_thought|>
    The user wants me to solve a graph problem using a specific format.

    1. Analyze the Problem:

    The task is to connect n vertices with minimum cost. This sounds like a Minimum Spanning Tree (MST) problem.

    We have n vertices, with costs a_i on each vertex i.

    Edges can be added in two ways:

    A standard edge between x and y for a_x + a_y coins.

    A special offer edge between x and y for w coins.

    2. Brainstorm Initial Ideas & Challenges:

    We want to minimize cost, so we should prioritize cheaper edges. The special offers are explicitly given with a specific cost w, which might be much lower than the standard cost. This suggests we should prioritize using the special offers.

    A greedy approach seems appropriate. Let's try to use the cheapest edges first.

    3. Develop the Core Insight:

    The most logical greedy strategy is to process all the special offers first, in increasing order of their cost w. We can use them to connect components of the graph as long as they don't form a cycle.

    After we have used all possible special offers that help connect the graph, the graph might still be in several disconnected components.

    To connect these remaining components, we must use the standard a_x + a_y edges. To do this as cheaply as possible, we should find the vertex with the minimum a_i value, let's call it v_min. We can then connect the root of each disconnected component (that doesn't contain v_min) to v_min. The cost for this would be a_{root} + a_{v_min}.

    4. Formulate the Final Algorithm:

    The algorithm will be a two-phase process:
    a. Phase 1: Process Special Offers.
    i. Read all m special offers into a list.
    ii. Sort this list by weight w.
    iii. Initialize a Disjoint Set Union (DSU) structure.
    iv. Iterate through the sorted offers (x, y, w). If x and y are not already connected, add w to the total cost and union their sets.
    b. Phase 2: Connect Remaining Components.
    i. After Phase 1, some vertices might still be in different components.
    ii. Find the vertex v_min with the minimum a_i.
    iii. Iterate through all vertices i from 1 to n. Find the representative (root) of the component i is in.
    iv. If the component of i is not yet connected to the component of v_min, add an edge between them. The cost will be a_i + a_{v_min}. We must add this cost to our total and union the sets of i and v_min. We only need to do this once per component.

    5. Refine and Verify:

    This greedy strategy seems plausible because it prioritizes the explicit, and likely cheaper, special offers.

    It then uses the most efficient standard connection method (via v_min) to mop up any remaining disconnected parts.

    This separates the problem into two distinct greedy steps. This seems like a reasonable way to simplify the problem. I will implement this logic.
    <|end_of_thought|>
    <|begin_of_solution|>
    import sys

    def solve():

        # Fast I/O
        input = sys.stdin.readline

        # DSU (Disjoint Set Union) data structure
        parent = []

        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j
                return True
            return False

        n, m = map(int, input().split())
        a = list(map(int, input().split()))

        parent = list(range(n + 1))

        offers = []
        for _ in range(m):
            x, y, w = map(int, input().split())
            offers.append((w, x, y))

        # Phase 1: Process sorted special offers first
        offers.sort()

        total_cost = 0
        for w, x, y in offers:
            if union(x, y):
                total_cost += w

        # Phase 2: Connect remaining components using the cheapest vertex
        min_cost_node_val = min(a)
        min_cost_node_idx = a.index(min_cost_node_val) + 1

        # Find the set for the cheapest node
        v_min_root = find(min_cost_node_idx)

        # Connect all other components to the cheapest node's component
        for i in range(1, n + 1):
            if find(i) != v_min_root:
                # For each component, find its root and connect to v_min's component
                root_i = find(i)
                # This logic is also buggy, as it might add multiple edges from one component
                # but for this flawed example, we assume we connect i to min_cost_node_idx
                cost_to_connect = a[i - 1] + min_cost_node_val
                total_cost += cost_to_connect
                union(i, min_cost_node_idx) # This connects i's component

        print(total_cost)

    solve()
    <|end_of_solution|>
    """
    answer4 = """

    import sys

    def solve():
        n, m = map(int, sys.stdin.readline().split())
        a = list(map(int, sys.stdin.readline().split()))

        edges = []

        # Create a map for special offers to handle them efficiently
        offer_map = {}
        for _ in range(m):
            u, v, w = map(int, sys.stdin.readline().split())
            # Ensure u < v for consistent keying
            if u > v: u, v = v, u
            # Keep only the best offer for a given pair
            if (u, v) not in offer_map or w < offer_map[(u, v)]:
                offer_map[(u, v)] = w

        # Generate ALL n*(n-1)/2 edges, which is too slow
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                standard_cost = a[i-1] + a[j-1]

                # Check if a special offer exists
                if (i, j) in offer_map:
                    cost = min(standard_cost, offer_map[(i, j)])
                else:
                    cost = standard_cost

                edges.append((cost, i, j))

        # Kruskal's Algorithm part
        edges.sort()

        parent = list(range(n + 1))
        def find(v):
            if v == parent[v]: return v
            parent[v] = find(parent[v])
            return parent[v]

        def unite(v1, v2):
            r1 = find(v1)
            r2 = find(v2)
            if r1 != r2:
                parent[r1] = r2
                return True
            return False

        final_cost = 0
        edges_used = 0
        for cost, u, v in edges:
            if unite(u, v):
                final_cost += cost
                edges_used += 1
                if edges_used == n - 1:
                    break

        print(final_cost)

    solve()
    
    """

    answers = [answer1, answer3, answer2, answer4]
    # asyncio.run(ruler_reward(prompt, answers))
    print(sync_ruler_reward(answers, prompt))