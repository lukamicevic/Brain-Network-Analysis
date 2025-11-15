load('M87141793_fiber.mat');
whos
% Visualize sparse fibergraph matrix

% Show sparsity pattern
figure;
spy(fibergraph);
title('Sparsity Pattern of fibergraph');

% Show heatmap
figure;
imagesc(full(fibergraph));
colorbar;
title('fibergraph Matrix Heatmap');

% Graph visualization
G = graph(fibergraph);
figure;
plot(G);
title('Graph Representation of fibergraph');

%% Load the file
load('c17e7483-8301-4c2a-8be6-4ed22b2309e1.mat');

%% Check matrix properties
disp('Size of fibergraph:');
disp(size(fibergraph));

disp('Is symmetric?');
disp(isequal(fibergraph, fibergraph'));

%% Visualize sparsity
figure;
spy(fibergraph);
title('Sparsity Pattern of fibergraph');

%% Visualize heatmap
figure;
imagesc(full(fibergraph));
colorbar;
title('fibergraph Heatmap');

%% If not symmetric, make a symmetric adjacency matrix
if ~isequal(fibergraph, fibergraph')
    A = fibergraph + fibergraph';   % symmetrize
else
    A = fibergraph;
end

%% Build and plot the graph
G = graph(A);
figure;
plot(G, 'Layout','force');
title('Symmetrized Fiber Graph (70 nodes)');
