package com.redhat.docbot;

import org.eclipse.microprofile.rest.client.annotation.ClientHeaderParam;
import org.eclipse.microprofile.rest.client.inject.RegisterRestClient;

import jakarta.ws.rs.GET;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.core.Response;

@RegisterRestClient(configKey = "kfp-client")
@ClientHeaderParam(name = "Authorization", value = "Bearer ${token}")
public interface KubeflowPipelineClient {

    @GET
    @Path("/apis/v2beta1/pipelines")
    Response getPipelines();

    @POST
    @Path("/apis/v2beta1/runs")
    Response runPipeline(PipelineSpec pipelineSpec);
}
