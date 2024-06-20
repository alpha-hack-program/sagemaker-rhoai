package com.redhat.docbot;

import java.util.Optional;

import javax.naming.ConfigurationException;

import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.ws.rs.core.Response;

import org.apache.camel.builder.RouteBuilder;
import org.apache.camel.component.aws2.s3.AWS2S3Constants;

import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.eclipse.microprofile.rest.client.inject.RestClient;

@ApplicationScoped 
public class S3EventListener extends RouteBuilder {

    @Inject
    @ConfigProperty(name = "bucket.name")
    String bucketName;

    @Inject
    @ConfigProperty(name = "kfp.pipeline.namespace")
    String kfpNamespace;

    @Inject
    @ConfigProperty(name = "kfp.pipeline.display-name")
    String pipelineDisplayName;

    @Inject
    @RestClient
    KubeflowPipelineClient kfpClient;

    private String pipelineId;

    @PostConstruct
    void init() throws ConfigurationException {
        try {
            fetchPipelineId();
        } catch (Exception e) {
            log.error("Failed to fetch pipeline ID", e);
            throw new ConfigurationException("Failed to fetch pipeline ID");
        }
    }

    @Override
    public void configure() throws Exception {
        from("aws2-s3://{{bucket.name}}?deleteAfterRead=true")
            .routeId("s3-event-listener")
            .log("Received S3 event: ${header.CamelAwsS3EventType}")
            .filter(header("CamelAwsS3Key").endsWith(".onnx"))
            .setHeader("CamelMinioObjectName", simple("${header.CamelAwsS3Key}")) 
            .log("Processing file: ${header.CamelMinioObjectName}")
            .to("minio://{{minio.bucket-name}}?accessKey={{minio.access-key}}&secretKey={{minio.secret-key}}&region={{minio.region}}&endpoint={{minio.endpoint}}")
            .process(exchange -> {
                // Log the S3 object key
                String key = exchange.getIn().getHeader(AWS2S3Constants.KEY, String.class);
                log.info("Processing file: " + key + " from bucket: " + bucketName);

                // Run the pipeline
                log.info("Running pipeline: " + pipelineDisplayName);
                runPipeline();
            });
    }

    private void fetchPipelineId() {
        Response response = kfpClient.getPipelines();
        if (response.getStatus() != 200) {
            throw new RuntimeException("Failed to fetch pipelines from Kubeflow");
        }

        PipelineResponse pipelineResponse = response.readEntity(PipelineResponse.class);
        log.info("Pipeline list: " + pipelineResponse.getPipelines());
        Optional<Pipeline> pipelineOpt = pipelineResponse.getPipelines().stream()
            .filter(p -> pipelineDisplayName.equals(p.getDisplayName()))
            .findFirst();

        if (pipelineOpt.isPresent()) {
            this.pipelineId = pipelineOpt.get().getPipelineId();
            log.info("Pipeline ID for display name " + pipelineDisplayName + ": " + this.pipelineId);
        } else {
            throw new RuntimeException("Pipeline with display name " + pipelineDisplayName + " not found");
        }
    }

    private void runPipeline() {
        PipelineSpec pipelineSpec = new PipelineSpec();
        pipelineSpec.setDisplayName(pipelineDisplayName + "_run");
        pipelineSpec.setDescription("This is run from Camel route");
        RuntimeConfig runtimeConfig = new RuntimeConfig();
        // runtimeConfig.setParameters(Map.of("param1", "value1"));
        pipelineSpec.setRuntimeConfig(runtimeConfig);
        PipelineVersionReference pipelineVersionReference = new PipelineVersionReference();
        pipelineVersionReference.setPipelineId(this.pipelineId);
        pipelineSpec.setPipelineVersionReference(pipelineVersionReference);

        Response response = kfpClient.runPipeline(pipelineSpec);
        if (response.getStatus() != 200) {
            throw new RuntimeException("Failed to run pipeline " + this.pipelineId + " in Kubeflow");
        }

        String stringResponse = response.readEntity(String.class);
        log.info("Pipeline Run: " +stringResponse);
    }
}
