#ifndef CAFFE_MULTILABEL_DATA_LAYER_HPP_
#define CAFFE_MULTILABEL_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/ldmark_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

template <typename Dtype>
class MultilabelDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit MultilabelDataLayer(const LayerParameter& param);
  virtual ~MultilabelDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "Data"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:

  virtual void load_batch(Batch<Dtype>* batch);

  LdmarkReader<Datum> reader_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYER_HPP_