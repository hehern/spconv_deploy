/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "onnx-parser.hpp"
#include "onnx/onnx-ml.pb.h"
#include "onnx/onnx-operators-ml.pb.h"
#include "node_add.hpp"
#include <fstream>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

namespace spconv{

#define LOGV(fmt, ...)                                         \
  do {                                                         \
    if (get_verbose()) {                                           \
      printf("\033[33m[Verb🚩]\033[0m " fmt "\n", __VA_ARGS__); \
    }                                                          \
  } while (0)

#define LOGERR(fmt, ...)                                        \
  do {                                                          \
    printf("\033[31m[Erro❌ ]\033[0m " fmt "\n", __VA_ARGS__);   \
  } while (0)

struct ParameterFP16Data{
    std::vector<unsigned short> data;
    std::vector<int> shape;
};

static onnx::TensorProto get_initializer(const onnx::GraphProto& graph, const std::string& name) {
    for (int i = 0; i < graph.initializer_size(); ++i) {
        auto& init = graph.initializer(i);
        if (init.name() == name) 
            return init;
    }
    LOGERR("Can not find initializer '%s' in ONNX.", name.c_str());
    return onnx::TensorProto();
};

static ParameterFP16Data get_initializer_data(const onnx::GraphProto& graph, const std::string& name) {
    auto proto = get_initializer(graph, name);
    if(proto.data_type() != onnx::TensorProto_DataType_FLOAT16){
        LOGERR("Can not support non float data type[%d] for initializer data.", proto.data_type());
        return ParameterFP16Data();
    }

    ParameterFP16Data output;
    output.shape.resize(proto.dims().size());
    std::transform(proto.dims().begin(), proto.dims().end(), output.shape.begin(), [](int64_t x){return (int)x;});

    size_t volumn = std::accumulate(output.shape.begin(), output.shape.end(), 1ul, std::multiplies<int64_t>());
    if(volumn * sizeof(unsigned short) != proto.raw_data().size()){
        LOGERR("Invalid parameter data size. %ld != %ld", volumn * sizeof(unsigned short), proto.raw_data().size());
        return ParameterFP16Data();
    }
    unsigned short* pdata = (unsigned short*)proto.raw_data().data();//因为cpu里没有fp16类型，unsigned int也占16bits，所以此处用unsigned int表示
    output.data = std::vector<unsigned short>(pdata, pdata + volumn);
    return output;
};

static onnx::AttributeProto get_attribute(const onnx::NodeProto& node, const std::string& name) {
    for (int i = 0; i < node.attribute_size(); ++i) {
        auto& attr = node.attribute(i);
        if (attr.name() == name) 
            return attr;
    }
    LOGV("Can not find attribute '%s' in node '%s', it will use the default value of.",
        name.c_str(), node.name().c_str());
    return onnx::AttributeProto();
};

static std::vector<int> get_attribute_as_intarray(const onnx::NodeProto& node, const std::string& name) {
    auto ints = get_attribute(node, name).ints();
    std::vector<int> output(ints.size());
    for (int i = 0; i < ints.size(); ++i) 
        output[i] = ints[i];
    return output;
};

std::shared_ptr<Engine> load_engine_from_onnx(const std::string& onnx_file, Precision precision, void* stream, bool mark_all_output){//加载onnx：lidar.backbone.xyz.onnx

    onnx::ModelProto model;
    std::fstream fin(onnx_file, std::ios::binary | std::ios::in);
    if (!model.ParseFromIstream(&fin)) {//反序列化onnx，读取模型
        LOGV("Parse onnx failed: %s", onnx_file.c_str());
        return nullptr;
    }

    auto builder = spconv::create_engine_builder();//?
    auto graph = model.graph();//graph，包含输入张量信息、输出张量信息、节点信息
    
    std::unordered_map<std::string, spconv::SparseDTensor*> tensor_map_by_name;//输入、输出tensor map
    for (int i = 0; i < graph.input_size(); ++i) {//遍历所有的输入张量,对于scn来讲输入只有一个
        auto name = graph.input(i).name();
        tensor_map_by_name[name] = builder->push_input(name);
    }

    // Add+ReLU 融合预处理: 统计每个 tensor 名字作为 node input 被引用的次数,
    // 以及 graph 直接输出的 tensor 名集合 —— 用于判定 Relu 能否安全下沉进 Add
    // (仅当 Add 的输出只被该 Relu 消费且不是 graph output 时才可融合,
    //  否则其他消费者会拿到 ReLU 之后的值而非原始 Add 结果)
    std::unordered_map<std::string, int> input_ref_count;
    std::unordered_set<std::string> graph_output_names;
    for (int i = 0; i < graph.node_size(); ++i) {
        auto& ref_node = graph.node(i);
        for (int j = 0; j < ref_node.input_size(); ++j) ++input_ref_count[ref_node.input(j)];
    }
    for (int i = 0; i < graph.output_size(); ++i) graph_output_names.insert(graph.output(i).name());
    std::unordered_map<std::string, spconv::Add*> add_output_map;//Add 输出名 -> Add 节点

    std::vector<spconv::SparseDTensor*> collect_outputs;
    for (int i = 0; i < model.graph().node_size(); ++i) {//遍历所有的node，有conv add relu等
        auto& node = model.graph().node(i);//取当前node
        if (node.op_type() == "SparseConvolution") {//是稀疏卷积（注意有两种稀疏卷积，type都是SparseConvolution，rulebook命名时候不一样）

            auto x = tensor_map_by_name[node.input(0)];//输入tensor
            auto weight = get_initializer_data(graph, node.input(1));//spconv0.weight
            auto bias   = get_initializer_data(graph, node.input(2));//spconv0.bias(可以通过netron打开onnx对应查看)
            auto weight_dynamic_ranges_proto = get_attribute(node, "weight_dynamic_ranges");//长度为16，具体含义未知！！！
            auto weight_dynamic_ranges = 
                std::vector<float>(weight_dynamic_ranges_proto.floats().begin(), weight_dynamic_ranges_proto.floats().end());//转换为vector类型

            auto n = builder->push_sparse_conv(
                node.name(), x, 
                get_attribute_as_intarray(node, "input_spatial_shape"),
                get_attribute_as_intarray(node, "output_spatial_shape"),
                weight.data, weight.shape,
                weight_dynamic_ranges,
                bias.data, bias.shape,
                get_attribute(node, "activation").s(),
                get_attribute_as_intarray(node, "kernel_size"),
                get_attribute_as_intarray(node, "stride"),
                get_attribute_as_intarray(node, "padding"),
                get_attribute_as_intarray(node, "dilation"),
                get_attribute(node, "input_dynamic_range").f(),
                get_attribute(node, "subm").i(),
                get_attribute(node, "output_bound").i(),
                get_attribute(node, "rulebook").s(),
                get_attribute(node, "precision").s() == "int8" ? Precision::Int8 : Precision::Float16,
                get_attribute(node, "output_precision").s() == "int8" ? Precision::Int8 : Precision::Float16,
                node.output(0)
            );

            if(mark_all_output){//false
                collect_outputs.push_back(n->output(0));
            }
            tensor_map_by_name[node.output(0)] = n->output(0);
        } else if (node.op_type() == "Add" || node.op_type() == "QuantAdd") {
            auto a = tensor_map_by_name[node.input(0)];
            auto b = tensor_map_by_name[node.input(1)];

            auto n = builder->push_add(
                node.name(),
                a, b,
                get_attribute(node, "input0_dynamic_range").f(),
                get_attribute(node, "input1_dynamic_range").f(),
                node.output(0),
                get_attribute(node, "precision").s() == "int8" ? Precision::Int8 : Precision::Float16,
                get_attribute(node, "output_precision").s() == "int8" ? Precision::Int8 : Precision::Float16
            );
            tensor_map_by_name[node.output(0)] = n->output(0);
            add_output_map[node.output(0)] = n;//记录 Add 输出名 -> 节点, 供 Relu 融合判定
        } else if (node.op_type() == "Relu") {
            // Add+ReLU 融合: 输入来自 Add 且该输出仅被本 Relu 消费、且非 graph output 时,
            // ReLU 下沉进 Add 节点 (output = max(0, a+b) 一次完成),
            // 并将 Relu 的输出名映射到 Add 的输出 tensor, 后续消费者拿到正确的拓扑
            auto fused_it = add_output_map.find(node.input(0));
            if (fused_it != add_output_map.end() &&
                input_ref_count[node.input(0)] == 1 &&
                graph_output_names.find(node.input(0)) == graph_output_names.end()) {
                fused_it->second->set_relu();
                tensor_map_by_name[node.output(0)] = fused_it->second->output(0);
                continue;
            }
            auto x = tensor_map_by_name[node.input(0)];
            auto n = builder->push_relu(node.name(), x, node.output(0));
            tensor_map_by_name[node.output(0)] = n->output(0);
        } else if (node.op_type() == "ScatterDense") {
            auto x = tensor_map_by_name[node.input(0)];
            auto input_spatial_shape = get_attribute_as_intarray(node, "input_spatial_shape");
            auto output_shape = get_attribute_as_intarray(node, "output_shape");
            auto format = get_attribute(node, "format").s();
            auto n = builder->push_dense(node.name(), x, format, node.output(0), input_spatial_shape, output_shape);
            tensor_map_by_name[node.output(0)] = n->output(0);
        } else if (node.op_type() == "Reshape") {
            auto x = tensor_map_by_name[node.input(0)];
            auto dims = get_attribute(node, "dims");
            std::vector<int64_t> shape(dims.ints().begin(), dims.ints().end());
            auto n = builder->push_reshape(node.name(), x, shape, node.output(0));
            tensor_map_by_name[node.output(0)] = n->output(0);
        } else if (node.op_type() == "Transpose") {
            auto x = tensor_map_by_name[node.input(0)];
            auto dims = get_attribute(node, "dims");
            std::vector<int64_t> shape(dims.ints().begin(), dims.ints().end());
            auto n = builder->push_transpose(node.name(), x, shape, node.output(0));
            tensor_map_by_name[node.output(0)] = n->output(0);
        } else {
            printf("Unsupport operator [%s]\b", node.op_type().c_str());
            return nullptr;
        }
    }

    for (int i = 0; i < graph.output_size(); ++i) {
        auto name = graph.output(i).name();
        collect_outputs.push_back(tensor_map_by_name[name]);
    }

    for (int i = 0; i < collect_outputs.size(); ++i) {
        builder->push_output(collect_outputs[i]);
    }
    return builder->build(precision, stream);//生成Engine并返回
}
};