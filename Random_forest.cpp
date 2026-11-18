#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
using namespace std;
vector<vector<double>>load_csv(const string& filename)
{
    
    vector<vector<double>> data;
    ifstream file(filename);
    if(!file.is_open())
    {
        cout<<"File is not available";
        return data;
    }
    string line="";
    while(getline(file,line))
    {
        if(line.empty()) continue;
        if(isalpha(line[0])) continue;
        vector<double> row;
        string cell;
        stringstream ss(line);

        while(getline(ss,cell,','))
        {
            row.push_back(stod(cell));
        }
        data.push_back(row);
    }
    file.close();
    return data;
}
double cal_gini(const vector<vector<double>>& rows)
{
    if(rows.empty()) 
    return 0.0;

    int c0=0,c1=0;
    for(const auto& row:rows)
    {
        if(row.back()==0)
        c0++;
        else
        c1++;
    }
    double total,p1,p2;
    total =rows.size();
    p1=c0/total;
    p2=c1/total;
    return 1-(p1*p1 + p2*p2);
}
void find_bs(
    const vector<vector<double>>& rows,
    int max_features,
    mt19937& gen,
    int& best_coln,
    double& best_tf,
    double& best_gini
)
{
    best_coln=-1;
    best_tf=0.0;
    best_gini=999.0;

    int fea_size=rows[0].size()-1;
    vector<int> col_indices(fea_size);
    for(int i=0;i<fea_size;i++)
    col_indices[i]=i;
    shuffle(col_indices.begin(),col_indices.end(),gen);
    int limit=min(fea_size,max_features);
    for(int i=0;i<limit;i++)
    {
        int col=col_indices[i];
        for(const auto& candidate_row:rows)
        {
            double threshold=candidate_row[col];

            vector<vector<double>> left_row;
            vector<vector<double>> right_row;
            for(const auto& row:rows)
            {
                if(row[col]<=threshold)
                left_row.push_back(row);
                else
                right_row.push_back(row);
            }

            if(left_row.empty()||right_row.empty()) continue; //skips empty

            double left_weight=(double)left_row.size()/rows.size();
            double right_weight=(double)right_row.size()/rows.size();
            double weight_gini=(left_weight*cal_gini(left_row))+(right_weight*cal_gini(right_row));

            if(weight_gini<best_gini)
            {
                best_gini=weight_gini;
                best_coln=col;
                best_tf=threshold;
            }
        }
    }
}
struct Node{
    int index=0;
    double threshold=0.0;
    int right_child=-1;
    int left_child=-1;
    int prediction=-1;
    bool is_leaf=false;
};
int major_class(const vector<vector<double>>& rows)
{
    int c0=0,c1=0;
    for(const auto& row:rows)
    if(row.back()==0.0)
    c0++;
    else
    c1++;
    return (c0>c1)?0:1;
}
int build_tree(const vector<vector<double>>& rows, int depth, int max_depth,
    int max_features,mt19937& gen, vector<Node>& tree)
{
    int current_idx=tree.size();
    tree.push_back(Node());

    if(depth>=max_depth||cal_gini(rows)==0.0||rows.size()<=1) //Base condition to stop the recursion
    {
        tree[current_idx].is_leaf=true;
        tree[current_idx].prediction=major_class(rows);
        return current_idx;
    }
    //finding best split and if nothing is found then turn that into leaf node
    int best_col;
    double best_tf,best_gini;
    find_bs(rows, max_features, gen, best_col, best_tf, best_gini);
    if(best_col==-1)
    {
        tree[current_idx].is_leaf=true;
        tree[current_idx].prediction=major_class(rows);
        return current_idx;
    }

    //diving the rows in left and right rows
    vector<vector<double>> right_row,left_row;
    for(const auto& row:rows)
    {
        if(row[best_col]<=best_tf)
        left_row.push_back(row);
        else
        right_row.push_back(row);
    }

    //saving the split rule and building the left and right tree
    tree[current_idx].index=best_col;
    tree[current_idx].threshold=best_tf;
    int left_idx = build_tree(left_row, depth + 1, max_depth,max_features,gen, tree);
    int right_idx = build_tree(right_row, depth + 1, max_depth,max_features,gen, tree);
    tree[current_idx].left_child = left_idx;
    tree[current_idx].right_child = right_idx;
    return current_idx;
}
int predict(const vector<Node>& tree, const vector<double>& rows)
{
    int curr=0;
    while(!tree[curr].is_leaf)
    {
        int col=tree[curr].index;
        double thresh=tree[curr].threshold;

        if(rows[col]<=thresh)
        curr=tree[curr].left_child;
        else
        curr=tree[curr].right_child;
    }
    return tree[curr].prediction;
}
vector<vector<double>> bootstrap_sample(const vector<vector<double>>& dataset, mt19937& gen)
{
    vector<vector<double>> sample;
    uniform_int_distribution<int> dist(0, dataset.size() - 1);

    for (size_t i = 0; i < dataset.size(); i++)
    {
        int random_idx = dist(gen);
        sample.push_back(dataset[random_idx]);
    }
    return sample;
}
vector<vector<Node>> train_forest(
    const vector<vector<double>>& dataset,
    int num_trees,
    int max_depth,
    mt19937& gen
)
{
    vector<vector<Node>> forest;
    int total_features = dataset[0].size() - 1;
    int max_features = max(1, (int)sqrt(total_features)); // Standard RF rule: sqrt(features)

    for (int t = 0; t < num_trees; t++)
    {
        vector<vector<double>> sample = bootstrap_sample(dataset, gen);
        vector<Node> tree;
        build_tree(sample, 0, max_depth, max_features, gen, tree);
        forest.push_back(tree);
    }
    return forest;
}
int predict(const vector<vector<Node>>& forest, const vector<double>& row)
{
    int votes_0 = 0;
    int votes_1 = 0;

    for (const auto& tree : forest)
    {
        int pred = predict(tree, row);
        if (pred == 0)
            votes_0++;
        else
            votes_1++;
    }

    return (votes_1 > votes_0) ? 1 : 0;
}
void test_train_split(vector<vector<double>>& dataset,
mt19937& gen,
double train_ratio,
vector<vector<double>>& train_data,
vector<vector<double>>& test_data)
{
    shuffle(dataset.begin(),dataset.end(),gen);
    int split_idx=(int)(dataset.size()*train_ratio);
    train_data.assign(dataset.begin(),dataset.begin()+split_idx);
    test_data.assign(dataset.begin()+split_idx,dataset.end());
}
double evaluate_accuracy(
    const vector<vector<Node>>& forest,
    const vector<vector<double>>& test_data
)
{
    if (test_data.empty()) return 0.0;

    int correct = 0;
    for (const auto& row : test_data)
    {
        int actual_label = (int)row.back();
        int predicted_label = predict(forest, row);

        if (predicted_label == actual_label)
            correct++;
    }

    return ((double)correct / test_data.size()) * 100.0;
}
int main()
{
    vector<vector<double>> dataset=load_csv("breast_cancer.csv");
    if(dataset.empty()) return 1;
    mt19937 gen(43);
    vector<vector<double>> train_data,test_data;
    test_train_split(dataset,gen,0.8,train_data,test_data);
    int num_trees = 50;
    int max_depth = 5;

    vector<vector<Node>> forest = train_forest(train_data, num_trees, max_depth, gen);
    cout << "Random Forest trained with " << forest.size() << " trees!\n";

    double accuracy=evaluate_accuracy(forest,test_data);
    cout<<"Accuracy is "<<accuracy<<endl;
    return 0;
}
