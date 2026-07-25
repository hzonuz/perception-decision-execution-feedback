"""Deterministic contract fixtures."""

from perception.contracts import (
    SCHEMA_VERSION,
    AgentNode,
    AgentState,
    AttributeValue,
    BernoulliBelief,
    BoundingBox2D,
    CameraCalibration,
    CameraExtrinsics,
    CameraIntrinsics,
    CaptureFrame,
    CategoricalDistribution,
    FrameStamp,
    ImageEncoding,
    ImageReference,
    LifecycleEvent,
    LifecycleEventType,
    MaskEncoding,
    MaskReference,
    MemoryState,
    NamedBelief,
    NormalizedPoint,
    ObjectAnnotation,
    ObjectNode,
    PrimitiveAction,
    Quaternion,
    Relation,
    RelationPredicate,
    SSGSnapshot,
    SourceMetadata,
    SourceType,
    Vector3,
)


def oracle_source() -> SourceMetadata:
    return SourceMetadata(
        source=SourceType.SIMULATOR_ORACLE,
        producer="perception_capture",
        producer_version="0.1.0",
        privileged=True,
        configuration_id="contracts_fixture",
    )


def fixture_stamp() -> FrameStamp:
    return FrameStamp(episode_id="fixture-episode-001", frame_index=0, timestamp_seconds=0.0)


def fixture_agent() -> AgentState:
    return AgentState(
        node_id="agent",
        world_position=Vector3(5.0, 0.5, 5.0),
        world_rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
        velocity=Vector3(0.0, 0.0, 0.0),
        recent_action=PrimitiveAction(forward=0, rotate=0, name="no_op"),
        health=100.0,
    )


def fixture_functional_types() -> tuple[NamedBelief, ...]:
    return (
        NamedBelief(
            name="collectable",
            belief=BernoulliBelief(probability=1.0, source=oracle_source()),
        ),
    )


def make_capture_frame() -> CaptureFrame:
    stamp = fixture_stamp()
    source = oracle_source()
    annotation = ObjectAnnotation(
        instance_id="episode-001:instance-0001",
        exact_simulator_class="GoodGoal",
        semantic_distribution=CategoricalDistribution.certain("positive_goal"),
        functional_types=fixture_functional_types(),
        world_position=Vector3(10.0, 1.5, 10.0),
        world_rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
        dimensions=Vector3(3.0, 3.0, 3.0),
        velocity=Vector3(0.0, 0.0, 0.0),
        bounding_box=BoundingBox2D(20.0, 18.0, 43.0, 49.0, 64, 64),
        instance_mask=MaskReference(
            path="masks/fixture-episode-001/000000/instance-0001.png",
            encoding=MaskEncoding.PNG,
            width=64,
            height=64,
            instance_value=1,
        ),
        relative_depth=7.25,
        visibility_fraction=0.9,
        occlusion_fraction=0.1,
        attributes=(
            AttributeValue("unity_tag", "goodGoal", 1.0, source),
        ),
        source=source,
    )
    return CaptureFrame(
        schema_version=SCHEMA_VERSION,
        episode_id=stamp.episode_id,
        arena_id="0",
        random_seed=7,
        stamp=stamp,
        rgb=ImageReference(
            path="rgb/fixture-episode-001/000000.png",
            width=64,
            height=64,
            encoding=ImageEncoding.PNG,
        ),
        camera=CameraCalibration(
            camera_id="AgentCam",
            intrinsics=CameraIntrinsics(
                width=64,
                height=64,
                fx=55.4256,
                fy=55.4256,
                cx=32.0,
                cy=32.0,
                near_clip=0.1,
                far_clip=1000.0,
            ),
            extrinsics=CameraExtrinsics(
                world_position=Vector3(5.0, 1.2, 5.0),
                world_rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
            ),
        ),
        agent=fixture_agent(),
        objects=(annotation,),
        lifecycle_events=(
            LifecycleEvent(
                instance_id=annotation.instance_id,
                event=LifecycleEventType.SPAWNED,
                stamp=stamp,
                reason="arena_reset",
                source=source,
            ),
        ),
        contact_events=(),
        source=source,
    )


def make_oracle_ssg() -> SSGSnapshot:
    stamp = fixture_stamp()
    source = oracle_source()
    agent = AgentNode(
        node_id="agent",
        state=fixture_agent(),
        stamp=stamp,
        confidence=1.0,
        source=source,
    )
    object_node = ObjectNode(
        node_id="obj_episode-001_instance-0001",
        track_id="episode-001:instance-0001",
        detector_id=None,
        exact_simulator_class="GoodGoal",
        semantic_distribution=CategoricalDistribution.certain("positive_goal"),
        functional_types=fixture_functional_types(),
        bounding_box=BoundingBox2D(20.0, 18.0, 43.0, 49.0, 64, 64),
        instance_mask=MaskReference(
            path="masks/fixture-episode-001/000000/instance-0001.png",
            encoding=MaskEncoding.PNG,
            width=64,
            height=64,
            instance_value=1,
        ),
        normalized_location=NormalizedPoint(0.4921875, 0.5234375),
        bearing_degrees=0.0,
        relative_distance=7.25,
        relative_velocity=Vector3(0.0, 0.0, 0.0),
        detector_confidence=None,
        visibility_probability=0.9,
        existence_probability=1.0,
        track_age_frames=1,
        last_seen=stamp,
        memory_state=MemoryState.VISIBLE,
        confidence=1.0,
        source=source,
        stamp=stamp,
        attributes=(AttributeValue("unity_tag", "goodGoal", 1.0, source),),
    )
    relation = Relation(
        subject_node_id=agent.node_id,
        predicate=RelationPredicate.NEAR,
        object_node_id=object_node.node_id,
        symbolic_value=None,
        probability=1.0,
        source=source,
        stamp=stamp,
        age_frames=0,
        evidence=("simulator_world_distance",),
    )
    return SSGSnapshot(
        schema_version=SCHEMA_VERSION,
        graph_id="fixture-episode-001:frame-000000",
        episode_id=stamp.episode_id,
        stamp=stamp,
        is_macro_step=True,
        agent=agent,
        objects=(object_node,),
        relations=(relation,),
        confidence=1.0,
        source=source,
    )
