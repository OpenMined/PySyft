import { UUID } from "./uid"

interface Parameter {
  _name: string
  // Other properties of Parameter go here
}

interface ReturnAnnotation {
  __args__?: string[]
  path: string
  // Other properties of ReturnAnnotation go here
}

/**
 * Represents a function's signature.
 */
class Signature {
  parameters: string[]
  return_annotation: ReturnAnnotation

  /**
   * Creates a new instance of Signature.
   * @param {Array} parameters An array of objects representing the function's parameters.
   * @param {Object} return_annotation An object representing the function's return annotation.
   */
  constructor(parameters: Parameter[], return_annotation: ReturnAnnotation) {
    this.parameters = parameters.map((param) => param._name)
    this.return_annotation = return_annotation
  }
}

interface InputObject {
  [key: string]: any
}

/**
 * Represents an input policy for a function.
 */
class InputPolicy {
  id: UUID
  inputs: Map<string, any>
  server_id?: UUID

  /**
   * Creates a new instance of InputPolicy.
   * @param {UUID} id The ID of the input policy.
   * @param {Object} inputs The inputs to the function.
   */
  constructor(id: UUID, inputs: InputObject) {
    this.id = id
    this.inputs = new Map(
      Object.entries(inputs).map(([key, value]) => [JSON.parse(key), value])
    )
  }
}

/**
 * Represents an output policy for a function.
 */
class OutputPolicy {
  id: UUID
  outputs: any[]
  state_type: string
  state: object

  /**
   * Creates a new instance of OutputPolicy.
   * @param {UUID} id The ID of the output policy.
   * @param {Array} outputs The function's output values.
   * @param {string} state_type The type of the function's state object.
   * @param {Object} output_state The function's state object.
   */
  constructor(
    id: UUID,
    outputs: any[],
    state_type: string,
    output_state: object
  ) {
    this.id = id
    this.outputs = outputs
    this.state_type = state_type
    this.state = output_state
  }
}

/**
 * Represents user code.
 */
export class UserCode {
  code_hash: string
  id: UUID
  input_policy: InputPolicy
  output_policy: OutputPolicy
  parsed_code: string
  raw_code: string
  service_func_name: string
  user_key: Uint8Array
  unique_func_name: string
  user_unique_func_name: string
  status: Map<string, any>
  signature: Signature

  /**
   * Creates a new instance of UserCode.
   * @param {Object} obj An object containing the user's code details.
   */
  constructor(obj: {
    code_hash: string
    id: UUID
    input_policy: { id: UUID; inputs: InputObject }
    output_policy: { id: UUID; outputs: any[]; state_type: string }
    output_policy_state: object
    parsed_code: string
    raw_code: string
    service_func_name: string
    user_verify_key: Uint8Array
    unique_func_name: string
    user_unique_func_name: string
    name: string
    status: { base_dict: { [key: string]: any } }
    signature: { parameters: Parameter[]; return_annotation: ReturnAnnotation }
  }) {
    this.code_hash = obj.code_hash
    this.id = obj.id
    this.input_policy = new InputPolicy(
      obj.input_policy.id,
      obj.input_policy.inputs
    )
    this.output_policy = new OutputPolicy(
      obj.output_policy.id,
      obj.output_policy.outputs,
      obj.output_policy.state_type,
      obj.output_policy_state
    )
    this.parsed_code = obj.parsed_code
    this.raw_code = obj.raw_code
    this.service_func_name = obj.service_func_name
    this.user_key = obj.user_verify_key
    this.unique_func_name = obj.unique_func_name
    this.user_unique_func_name = obj.user_unique_func_name
    const newStatus = new Map(
      Object.entries(obj.status.base_dict).map(([key, value]) => [
        JSON.parse(key),
        value,
      ])
    )
    this.status = newStatus
    this.signature = new Signature(
      obj.signature.parameters,
      obj.signature.return_annotation
    )
  }
}
