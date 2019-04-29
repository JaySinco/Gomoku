#include <iomanip>

#include "network.h"

#define MX_TRY \
  try {
#define MX_CATCH \
  } catch(dmlc::Error &err) { \
    std::cout << MXGetLastError() << std::endl; \
    std::exit(-1); \
  }

using namespace mxnet::cpp;

void SampleData::flip_verticing() {
    for (int row = 0; row < BOARD_MAX_ROW; ++row) {
        for (int col = 0; col < BOARD_MAX_COL / 2; ++col) {
            int a = row * BOARD_MAX_COL + col;
            int b = row * BOARD_MAX_COL + BOARD_MAX_COL - col - 1;
            std::iter_swap(data + a, data + b);
            std::iter_swap(data + BOARD_SIZE + a, data + BOARD_SIZE + b);
            if (INPUT_FEATURE_NUM > 2)
                std::iter_swap(data + 2 * BOARD_SIZE + a, data + 2 * BOARD_SIZE + b);
            std::iter_swap(p_label + a, p_label + b);
        }
    }
}

void SampleData::transpose() {
    for (int row = 0; row < BOARD_MAX_ROW; ++row) {
        for (int col = row + 1; col < BOARD_MAX_COL; ++col) {
            int a = row * BOARD_MAX_COL + col;
            int b = col * BOARD_MAX_COL + row;
            std::iter_swap(data + a, data + b);
            std::iter_swap(data + BOARD_SIZE + a, data + BOARD_SIZE + b);
            if (INPUT_FEATURE_NUM > 2)
                std::iter_swap(data + 2 * BOARD_SIZE + a, data + 2 * BOARD_SIZE + b);
            std::iter_swap(p_label + a, p_label + b);
        }
    }
}

std::ostream &operator<<(std::ostream &out, const SampleData &sample) {
    Move last(NO_MOVE_YET);
    float first = -1.0f;
    for (int row = 0; row < BOARD_MAX_ROW; ++row) {
        for (int col = 0; col < BOARD_MAX_COL; ++col) {
            if (sample.data[row * BOARD_MAX_COL + col] > 0)
                out << Color::Black;
            else if (sample.data[BOARD_SIZE + row * BOARD_MAX_COL + col] > 0)
                out << Color::White;
            else
                out << Color::Empty;
            if (INPUT_FEATURE_NUM > 2) {
                if (sample.data[2 * BOARD_SIZE + row * BOARD_MAX_COL + col] > 0) {
                    assert(last.z() == NO_MOVE_YET);
                    last = Move(row, col);
                }
            }
            if (INPUT_FEATURE_NUM > 3) {
                if (first < 0)
                    first = sample.data[3 * BOARD_SIZE + row * BOARD_MAX_COL + col];
                else
                    assert(first == sample.data[3 * BOARD_SIZE + row * BOARD_MAX_COL + col]);
            }
        }
        out << "｜";
        for (int col = 0; col < BOARD_MAX_COL; ++col)
            out << " " << std::setw(5) << std::fixed << std::setprecision(1)
                << sample.p_label[row * BOARD_MAX_COL + col] * 100 << "%,";
        out << std::endl;
    }
    out << "↑value=" << sample.v_label[0];
    if (INPUT_FEATURE_NUM > 2) {
        out << ", last_move=";
        if (last.z() == NO_MOVE_YET)
            out << "None";
        else
            out << last;
    }
    if (INPUT_FEATURE_NUM > 3)
        out << ", fist_hand=" << first;
    out << std::endl;
    return out;
}

std::ostream &operator<<(std::ostream &out, const MiniBatch &batch) {
    for (int i = 0; i < BATCH_SIZE; ++i) {
        SampleData item;
        std::copy(batch.data + i * INPUT_FEATURE_NUM * BOARD_SIZE,
            batch.data + (i + 1) * INPUT_FEATURE_NUM * BOARD_SIZE, item.data);
        std::copy(batch.p_label + i * BOARD_SIZE, batch.p_label + (i + 1) * BOARD_SIZE, item.p_label);
        std::copy(batch.v_label + i, batch.v_label + (i + 1), item.v_label);
        out << item << std::endl;
    }
    return out;
}

void DataSet::push_with_transform(SampleData *data) {
    for (int i = 0; i < 4; ++i) {
        data->transpose();
        push_back(data);
        data->flip_verticing();
        push_back(data);
    }
}

void DataSet::make_mini_batch(MiniBatch *batch) const {
    assert(index > BATCH_SIZE);
    std::uniform_int_distribution<int> uniform(0, size() - 1);
    for (int i = 0; i < BATCH_SIZE; i++) {
        int c = uniform(global_random_engine);
        SampleData *r = buf + c;
        std::copy(std::begin(r->data), std::end(r->data), batch->data + INPUT_FEATURE_NUM * BOARD_SIZE * i);
        std::copy(std::begin(r->p_label), std::end(r->p_label), batch->p_label + BOARD_SIZE * i);
        std::copy(std::begin(r->v_label), std::end(r->v_label), batch->v_label + i);
    }
}

std::ostream &operator<<(std::ostream &out, const DataSet &set) {
    for (int i = 0; i < set.size(); ++i)
        out << set.get(i) << std::endl;
    return out;
}

Symbol dense_layer(const std::string &name, Symbol data,
        int num_hidden, const std::string &act_type) {
    Symbol w(name + "_w"), b(name + "_b");
    Symbol out = FullyConnected("fc_" + name, data, w, b, num_hidden);
    if (act_type != "None")
        out = Activation("act_" + name, out, act_type);
    return out;
}

Symbol convolution_layer(const std::string &name, Symbol data,
        int num_filter, Shape kernel, Shape stride, Shape pad,
        bool use_act, bool use_bn = USE_BATCH_NORM) {
    Symbol conv_w(name + "_w");
    Symbol conv_b(name + "_b");
    Symbol out = Convolution("conv_" + name, data,
        conv_w, conv_b, kernel, num_filter, stride, Shape(1, 1), pad);
    if (use_bn) {
        Symbol gamma(name + "_bn_gamma");
        Symbol beta(name + "_bn_beta");
        Symbol mmean(name + "_bn_mmean");
        Symbol mvar(name + "_bn_mvar");
        out = BatchNorm("bn_" + name, out, gamma, beta, mmean, mvar);
    }
    if (use_act)
        out = Activation("relu_" + name, out, "relu");
    return out;
}

Symbol residual_layer(const std::string &name, Symbol data,
        int num_filter) {
    Symbol conv1 = convolution_layer(name + "_conv1", data, num_filter,
        Shape(3, 3), Shape(1, 1), Shape(1, 1), true);
    Symbol conv2 = convolution_layer(name + "_conv2", conv1, num_filter,
        Shape(3, 3), Shape(1, 1), Shape(1, 1), false);
    return Activation("relu_" + name, data + conv2, "relu");
}

Symbol residual_block(const std::string &name, Symbol data,
        int num_block, int num_filter) {
    Symbol out = data;
    for (int i = 0; i < num_block; ++i)
        out = residual_layer(name + "_block" + std::to_string(i + 1), out, num_filter);
    return out;
}

Symbol middle_layer(Symbol data) {
    Symbol middle_conv = convolution_layer("middle_conv", data,
        NET_NUM_FILTER, Shape(3, 3), Shape(1, 1), Shape(1, 1), true);
    Symbol middle_residual = residual_block("middle_res", middle_conv, NET_NUM_RESIDUAL_BLOCK, NET_NUM_FILTER);
    return middle_residual;
}

std::pair<Symbol, Symbol> plc_layer(Symbol data, Symbol label) {
    Symbol plc_conv = convolution_layer("plc_conv", data,
        2, Shape(1, 1), Shape(1, 1), Shape(0, 0), true);
    Symbol plc_logist_out = dense_layer("plc_logist_out", plc_conv, BOARD_SIZE, "None");
    Symbol plc_out = softmax("plc_out", plc_logist_out);
    Symbol plc_m_loss = -1 * elemwise_mul(label, log_softmax(plc_logist_out));
    Symbol plc_loss = MakeLoss(mean(sum(plc_m_loss, dmlc::optional<Shape>(Shape(1)))));
    return std::make_pair(plc_out, plc_loss);
}

std::pair<Symbol, Symbol> val_layer(Symbol data, Symbol label) {
    Symbol val_conv = convolution_layer("val_conv", data,
        1, Shape(1, 1), Shape(1, 1), Shape(0, 0), true);
    Symbol val_dense = dense_layer("val_dense", val_conv, NET_NUM_FILTER, "relu");
    Symbol val_out = dense_layer("val_logist_out", val_dense, 1, "tanh");
    Symbol val_loss = MakeLoss(mean(square(elemwise_sub(val_out, label))));
    return std::make_pair(val_out, val_loss);
}

FIRNet::FIRNet(long long verno) : update_cnt(verno), ctx(Context::cpu()),
        data_predict(NDArray(Shape(1, INPUT_FEATURE_NUM, BOARD_MAX_ROW, BOARD_MAX_COL), ctx)),
        data_train(NDArray(Shape(BATCH_SIZE, INPUT_FEATURE_NUM, BOARD_MAX_ROW, BOARD_MAX_COL), ctx)),
        plc_label(NDArray(Shape(BATCH_SIZE, BOARD_SIZE), ctx)),
        val_label(NDArray(Shape(BATCH_SIZE, 1), ctx)) {
    MX_TRY
    build_graph();
    if (update_cnt > 0)
        load_param();
    bind_train();
    if (update_cnt == 0) {
        loss.InferArgsMap(ctx, &args_map, args_map);
        auxs_map = loss_train->aux_dict();
        init_param();
    }
    bind_predict();
    optimizer = OptimizerRegistry::Find("sgd");
    optimizer->SetParam("momentum", 0.9)
        ->SetParam("clip_gradient", 10)
        ->SetParam("lr", calc_init_lr())
        ->SetParam("wd", WEIGHT_DECAY);
    MX_CATCH
}

float FIRNet::calc_init_lr() {
    float multiplier;
    if (update_cnt < LR_DROP_STEP1)
        multiplier = 1.0f;
    else if (update_cnt >= LR_DROP_STEP1 && update_cnt < LR_DROP_STEP2)
        multiplier = 1e-1;
    else if (update_cnt >= LR_DROP_STEP2 && update_cnt < LR_DROP_STEP3)
        multiplier = 1e-2;
    else
        multiplier = 1e-3;
    float lr = INIT_LEARNING_RATE  * multiplier;
    LOG(INFO) << "init learning_rate=" << lr;
    return lr;
}

void FIRNet::adjust_lr() {
    float multiplier = 1.0f;
    switch (update_cnt) {
    case LR_DROP_STEP1: multiplier = 1e-1; break;
    case LR_DROP_STEP2: multiplier = 1e-2; break;
    case LR_DROP_STEP3: multiplier = 1e-3; break;
    }
    if (multiplier < 1.0f) {
        float lr = INIT_LEARNING_RATE  * multiplier;
        optimizer->SetParam("lr", lr);
        LOG(INFO) << "adjusted learning_rate=" << lr;
    }
}

FIRNet::~FIRNet() {
    delete plc_predict;
    delete val_predict;
    delete loss_train;
    delete optimizer;
    //MXNotifyShutdown();
}

void FIRNet::build_graph() {
    auto middle = middle_layer(Symbol::Variable("data"));
    auto plc_pair = plc_layer(middle, Symbol::Variable("plc_label"));
    auto val_pair = val_layer(middle, Symbol::Variable("val_label"));
    plc = plc_pair.first;
    val = val_pair.first;
    loss = plc_pair.second + val_pair.second;
    loss_arg_names = loss.ListArguments();
}

void FIRNet::bind_train() {
    args_map["data"] = data_train;
    args_map["plc_label"] = plc_label;
    args_map["val_label"] = val_label;
    loss_train = loss.SimpleBind(ctx, args_map,
        std::map<std::string, NDArray>(),
        std::map<std::string, OpReqType>(),
        auxs_map);
}

void FIRNet::bind_predict() {
    args_map["data"] = data_predict;
    plc_predict = plc.SimpleBind(ctx, args_map,
        std::map<std::string, NDArray>(),
        std::map<std::string, OpReqType>(),
        auxs_map);
    val_predict = val.SimpleBind(ctx, args_map,
        std::map<std::string, NDArray>(),
        std::map<std::string, OpReqType>(),
        auxs_map);
    args_map.erase("data");
    args_map.erase("plc_label");
    args_map.erase("val_label");
}

void FIRNet::init_param() {
    auto xavier_init = Xavier(Xavier::gaussian, Xavier::in, 2.34);
    for (auto &arg : args_map) {
        xavier_init(arg.first, &arg.second);
    }
    auto mean_init = Constant(0.0f);
    auto mvar_init = Constant(BN_MVAR_INIT);
    for (auto &aux : auxs_map) {
        if (aux.first.find("_bn_mmean") != -1)
            mean_init(aux.first, &aux.second);
        else if (aux.first.find("_bn_mvar") != -1)
            mvar_init(aux.first, &aux.second);
    }
}

std::string FIRNet::make_param_file_name() {
    std::ostringstream filename;
    filename << "FIR-" << BOARD_MAX_COL << "x" << NET_NUM_FILTER
        << "i" << NET_NUM_RESIDUAL_BLOCK << "@" << update_cnt << ".param";
    return filename.str();
}

void FIRNet::load_param() {
    MX_TRY
    auto file_name = make_param_file_name();
    LOG(INFO) << "loading parameters from " << file_name;
    std::map<std::string, NDArray> param_map;
    NDArray::Load(file_name, nullptr, &param_map);
    for (const auto &param : param_map) {
        if (param.first.size() > 5 && param.first.substr(0, 5) == "_AUX_")
            auxs_map.insert(std::make_pair(param.first.substr(5), param.second));
        else
            args_map.insert(std::make_pair(param.first, param.second));
    }
    MX_CATCH
}

void FIRNet::save_param() {
    MX_TRY
    auto file_name = make_param_file_name();
    LOG(INFO) << "saving parameters into " << file_name;
    std::map<std::string, NDArray> param_map(args_map);
    for (const auto &aux : auxs_map)
        param_map.insert(std::make_pair("_AUX_" + aux.first, aux.second));
    NDArray::Save(file_name, param_map);
    MX_CATCH
}

void brief_NDArray(std::ostream &out, const std::string &name, const NDArray &nd) {
    out << std::left << std::setw(40) << name << " (";
    auto shape = nd.GetShape();
    for (int i = 0; i < shape.size(); ++i) {
        out << shape[i];
        if (i != shape.size() - 1)
            out << ", ";
    }
    out << ") = [";
    auto data = nd.GetData();
    auto num = nd.Size() > 3 ? 3 : nd.Size();
    for (int i = 0; i < num; ++i) {
        out << data[i];
        if (i != num - 1)
            out << ", ";
    }
    if (num < nd.Size())
        out << "...";
    out << "]\n";
}

void FIRNet::show_param(std::ostream &out) {
    out << "=== trainable parameters ===\n";
    for (const auto &arg : args_map)
        brief_NDArray(out, arg.first, arg.second);
    out << "\n=== auxiliary parameters ===\n";
    for (const auto &aux : auxs_map)
        brief_NDArray(out, aux.first, aux.second);
}

void mapping_data(int id, float data[INPUT_FEATURE_NUM * BOARD_SIZE]) {
    int n = 0;
    while (true) {
        if (n == id) break;
        // transpose
        for (int row = 0; row < BOARD_MAX_ROW; ++row) {
            for (int col = row + 1; col < BOARD_MAX_COL; ++col) {
                int a = row * BOARD_MAX_COL + col;
                int b = col * BOARD_MAX_COL + row;
                std::iter_swap(data + a, data + b);
                std::iter_swap(data + BOARD_SIZE + a, data + BOARD_SIZE + b);
                if (INPUT_FEATURE_NUM > 2)
                    std::iter_swap(data + 2 * BOARD_SIZE + a, data + 2 * BOARD_SIZE + b);
            }
        }
        ++n;
        if (n == id) break;
        // flip_verticing
        for (int row = 0; row < BOARD_MAX_ROW; ++row) {
            for (int col = 0; col < BOARD_MAX_COL / 2; ++col) {
                int a = row * BOARD_MAX_COL + col;
                int b = row * BOARD_MAX_COL + BOARD_MAX_COL - col - 1;
                std::iter_swap(data + a, data + b);
                std::iter_swap(data + BOARD_SIZE + a, data + BOARD_SIZE + b);
                if (INPUT_FEATURE_NUM > 2)
                    std::iter_swap(data + 2 * BOARD_SIZE + a, data + 2 * BOARD_SIZE + b);
            }
        }
        ++n;
    }
}

Move mapping_move(int id, Move mv) {
    int n = 0, r, c;
    while (true) {
        if (n == id) break;
        // transpose
        r = mv.c(), c = mv.r();
        mv = Move(r, c);
        ++n;
        if (n == id) break;
        // flip_verticing
        r = mv.r(), c = BOARD_MAX_COL - mv.c() - 1;
        mv = Move(r, c);
        ++n;
    }
    return mv;
}

void FIRNet::forward(const State &state,
        float value[1], std::vector<std::pair<Move, float>> &net_move_priors) {
    MX_TRY
    float data[INPUT_FEATURE_NUM * BOARD_SIZE] = { 0.0f };
    state.fill_feature_array(data);
    std::uniform_int_distribution<int> uniform(0, 7);
    int transform_id = uniform(global_random_engine);
    mapping_data(transform_id, data);
    data_predict.SyncCopyFromCPU(data, INPUT_FEATURE_NUM * BOARD_SIZE);
    plc_predict->Forward(false);
    val_predict->Forward(false);
    NDArray::WaitAll();
    const float *plc_ptr = plc_predict->outputs[0].GetData();
    float priors_sum = 0.0f;
    for (const auto mv : state.get_options()) {
        Move mapped = mapping_move(transform_id, mv);
        float prior = plc_ptr[mapped.z()];
        net_move_priors.push_back(std::make_pair(mv, prior));
        priors_sum += prior;
    }
    if (priors_sum < 1e-8) {
        LOG(INFO) << "wield policy probality yield by network: sum=" << priors_sum
            << ", available_move_n=" << net_move_priors.size();
        for (auto &item : net_move_priors)
            item.second = 1.0f / float(net_move_priors.size());
    }
    else {
        for (auto &item : net_move_priors)
            item.second /= priors_sum;
    }
    value[0] = val_predict->outputs[0].GetData()[0];
    MX_CATCH
}

float FIRNet::train_step(const MiniBatch *batch) {
    MX_TRY
    data_train.SyncCopyFromCPU(batch->data, BATCH_SIZE * INPUT_FEATURE_NUM * BOARD_SIZE);
    plc_label.SyncCopyFromCPU(batch->p_label, BATCH_SIZE * BOARD_SIZE);
    val_label.SyncCopyFromCPU(batch->v_label, BATCH_SIZE);
    loss_train->Forward(true);
    loss_train->Backward();
    for (int i = 0; i < loss_arg_names.size(); ++i) {
        if (loss_arg_names[i] == "data" || loss_arg_names[i] == "plc_label" ||
            loss_arg_names[i] == "val_label")
            continue;
        optimizer->Update(i, loss_train->arg_arrays[i], loss_train->grad_arrays[i]);
    }
    ++update_cnt;
    adjust_lr();
    NDArray::WaitAll();
    return loss_train->outputs[0].GetData()[0];
    MX_CATCH
}