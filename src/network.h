#pragma once

#include <mxnet-cpp/MxNetCpp.h>

#include "game.h"

struct SampleData {
    float data[INPUT_FEATURE_NUM * BOARD_SIZE] = { 0.0f };
    float p_label[BOARD_SIZE] = { 0.0f };
    float v_label[1] = { 0.0f };

    void flip_verticing();
    void transpose();
};
std::ostream &operator<<(std::ostream &out, const SampleData &sample);

struct MiniBatch {
    float data[BATCH_SIZE * INPUT_FEATURE_NUM * BOARD_SIZE] = { 0.0f };
    float p_label[BATCH_SIZE * BOARD_SIZE] = { 0.0f };
    float v_label[BATCH_SIZE * 1] = { 0.0f };
};
std::ostream &operator<<(std::ostream &out, const MiniBatch &batch);

class DataSet {
private:
    long long index;
    SampleData *buf;
public:
    DataSet() : index(0) { buf = new SampleData[BUFFER_SIZE]; }
    ~DataSet() { delete [] buf; }
    int size() const { return (index > BUFFER_SIZE) ? BUFFER_SIZE : index; }
    long long total() const { return index; }
    void push_back(const SampleData *data) { buf[index % BUFFER_SIZE] = *data; ++index; }
    void push_with_transform(SampleData *data);
    const SampleData &get(int i) const { assert(i < size()); return buf[i]; }
    void make_mini_batch(MiniBatch *batch) const;
};
std::ostream &operator<<(std::ostream &out, const DataSet &set);

class FIRNet {
    using Symbol = mxnet::cpp::Symbol;
    using Context = mxnet::cpp::Context;
    using NDArray = mxnet::cpp::NDArray;
    using Executor = mxnet::cpp::Executor;
    using Optimizer = mxnet::cpp::Optimizer;

    const Context ctx;
    std::map<std::string, NDArray> args_map;
    std::map<std::string, NDArray> auxs_map;
    std::vector<std::string> loss_arg_names;
    Symbol plc, val, loss;
    NDArray data_predict, data_train, plc_label, val_label;
    Executor *plc_predict, *val_predict, *loss_train;
    Optimizer* optimizer;
    long long update_cnt;
public:
    FIRNet(long long verno);
    ~FIRNet();
    long long verno() { return update_cnt; }
    void init_param();
    void save_param();
    void load_param();
    void show_param(std::ostream &out);
    void build_graph();
    void bind_train();
    void bind_predict();
    float calc_init_lr();
    void adjust_lr();
    std::string make_param_file_name();
    float train_step(const MiniBatch *batch);
    void forward(const State &state,
        float value[1], std::vector<std::pair<Move, float>> &move_priors);
};
