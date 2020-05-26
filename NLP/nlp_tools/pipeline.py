from consecution import Pipeline, GlobalState


class NLPPipeline(Pipeline):
    def consume(self, data_dict):
        self.global_state = GlobalState(processing_data=data_dict)
        self.begin()
        self.top_node._process(data_dict)
        return self.end()

    def end(self):
        return self.global_state.processing_data
