/*
 * poker_binding.c - Python C extension for Texas Hold'em Poker
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "poker.h"

// ============================================================================
// Batch Environment Python Object
// ============================================================================

typedef struct {
    PyObject_HEAD
    PokerBatchEnv batch;
    PyArrayObject* obs_array;
    PyArrayObject* reward_array;
    PyArrayObject* terminal_array;
    PyArrayObject* truncation_array;
} PokerBatchEnvObject;

static void PokerBatchEnvObject_dealloc(PokerBatchEnvObject* self) {
    batch_env_free(&self->batch);
    Py_XDECREF(self->obs_array);
    Py_XDECREF(self->reward_array);
    Py_XDECREF(self->terminal_array);
    Py_XDECREF(self->truncation_array);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PokerBatchEnvObject_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    PokerBatchEnvObject* self = (PokerBatchEnvObject*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->batch.envs = NULL;
        self->obs_array = NULL;
        self->reward_array = NULL;
        self->terminal_array = NULL;
        self->truncation_array = NULL;
    }
    return (PyObject*)self;
}

static int PokerBatchEnvObject_init(PokerBatchEnvObject* self, PyObject* args, PyObject* kwds) {
    static char* kwlist[] = {"num_envs", "num_players", "small_blind", "big_blind",
                             "starting_stack", "seed", NULL};

    int num_envs = 1;
    int num_players = 6;
    float small_blind = 0.5f;
    float big_blind = 1.0f;
    float starting_stack = 100.0f;
    unsigned long long seed = 42;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iifffK", kwlist,
                                     &num_envs, &num_players, &small_blind,
                                     &big_blind, &starting_stack, &seed)) {
        return -1;
    }

    if (num_players < 2 || num_players > MAX_PLAYERS) {
        PyErr_SetString(PyExc_ValueError, "num_players must be 2-6");
        return -1;
    }

    // Initialize batch environment
    batch_env_init(&self->batch, num_envs, num_players, small_blind,
                   big_blind, starting_stack, (uint64_t)seed);

    // Create numpy arrays
    npy_intp obs_dims[2] = {num_envs, OBS_SIZE};
    npy_intp vec_dims[1] = {num_envs};

    self->obs_array = (PyArrayObject*)PyArray_ZEROS(2, obs_dims, NPY_FLOAT32, 0);
    self->reward_array = (PyArrayObject*)PyArray_ZEROS(1, vec_dims, NPY_FLOAT32, 0);
    self->terminal_array = (PyArrayObject*)PyArray_ZEROS(1, vec_dims, NPY_UINT8, 0);
    self->truncation_array = (PyArrayObject*)PyArray_ZEROS(1, vec_dims, NPY_UINT8, 0);

    if (!self->obs_array || !self->reward_array ||
        !self->terminal_array || !self->truncation_array) {
        return -1;
    }

    // Set buffer pointers
    batch_env_set_buffers(&self->batch,
        (float*)PyArray_DATA(self->obs_array),
        (float*)PyArray_DATA(self->reward_array),
        (uint8_t*)PyArray_DATA(self->terminal_array),
        (uint8_t*)PyArray_DATA(self->truncation_array));

    return 0;
}

static PyObject* PokerBatchEnvObject_reset(PokerBatchEnvObject* self, PyObject* args) {
    batch_env_reset(&self->batch);

    PyObject* info = PyDict_New();
    Py_INCREF(self->obs_array);
    return Py_BuildValue("(OO)", self->obs_array, info);
}

static PyObject* PokerBatchEnvObject_step(PokerBatchEnvObject* self, PyObject* args) {
    PyArrayObject* actions_array;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &actions_array)) {
        return NULL;
    }

    if (PyArray_NDIM(actions_array) != 1 ||
        PyArray_DIM(actions_array, 0) != self->batch.num_envs) {
        PyErr_SetString(PyExc_ValueError, "actions must be 1D array with num_envs elements");
        return NULL;
    }

    PyArrayObject* actions_int = (PyArrayObject*)PyArray_Cast(actions_array, NPY_INT32);
    if (!actions_int) return NULL;

    // Create effective_actions array
    npy_intp dims[1] = {self->batch.num_envs};
    PyArrayObject* effective = (PyArrayObject*)PyArray_ZEROS(1, dims, NPY_INT32, 0);

    int* actions = (int*)PyArray_DATA(actions_int);
    int* eff = (int*)PyArray_DATA(effective);
    batch_env_step(&self->batch, actions, eff);

    Py_DECREF(actions_int);

    PyObject* info = PyDict_New();
    PyDict_SetItemString(info, "effective_actions", (PyObject*)effective);
    Py_DECREF(effective);

    Py_INCREF(self->obs_array);
    Py_INCREF(self->reward_array);
    Py_INCREF(self->terminal_array);
    Py_INCREF(self->truncation_array);

    return Py_BuildValue("(OOOOO)",
        self->obs_array,
        self->reward_array,
        self->terminal_array,
        self->truncation_array,
        info);
}

static PyObject* PokerBatchEnvObject_set_buffers(PokerBatchEnvObject* self, PyObject* args) {
    PyArrayObject *obs, *rewards, *terminals, *truncations;

    if (!PyArg_ParseTuple(args, "O!O!O!O!",
            &PyArray_Type, &obs,
            &PyArray_Type, &rewards,
            &PyArray_Type, &terminals,
            &PyArray_Type, &truncations)) {
        return NULL;
    }

    if (PyArray_NDIM(obs) != 2 || PyArray_DIM(obs, 0) != self->batch.num_envs ||
        PyArray_DIM(obs, 1) != OBS_SIZE) {
        PyErr_SetString(PyExc_ValueError, "obs shape mismatch");
        return NULL;
    }

    Py_XDECREF(self->obs_array);
    Py_XDECREF(self->reward_array);
    Py_XDECREF(self->terminal_array);
    Py_XDECREF(self->truncation_array);

    Py_INCREF(obs);
    Py_INCREF(rewards);
    Py_INCREF(terminals);
    Py_INCREF(truncations);

    self->obs_array = obs;
    self->reward_array = rewards;
    self->terminal_array = terminals;
    self->truncation_array = truncations;

    batch_env_set_buffers(&self->batch,
        (float*)PyArray_DATA(obs),
        (float*)PyArray_DATA(rewards),
        (uint8_t*)PyArray_DATA(terminals),
        (uint8_t*)PyArray_DATA(truncations));

    Py_RETURN_NONE;
}

static PyObject* PokerBatchEnvObject_get_num_envs(PokerBatchEnvObject* self, void* closure) {
    return PyLong_FromLong(self->batch.num_envs);
}

static PyObject* PokerBatchEnvObject_get_num_players(PokerBatchEnvObject* self, void* closure) {
    return PyLong_FromLong(self->batch.num_players);
}

static PyGetSetDef PokerBatchEnvObject_getsetters[] = {
    {"num_envs", (getter)PokerBatchEnvObject_get_num_envs, NULL, "Number of environments", NULL},
    {"num_players", (getter)PokerBatchEnvObject_get_num_players, NULL, "Number of players", NULL},
    {NULL}
};

static PyMethodDef PokerBatchEnvObject_methods[] = {
    {"reset", (PyCFunction)PokerBatchEnvObject_reset, METH_NOARGS, "Reset all environments"},
    {"step", (PyCFunction)PokerBatchEnvObject_step, METH_VARARGS, "Take a step"},
    {"set_buffers", (PyCFunction)PokerBatchEnvObject_set_buffers, METH_VARARGS, "Set external buffers"},
    {NULL}
};

static PyTypeObject PokerBatchEnvType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "poker_c.PokerBatchEnv",
    .tp_doc = "Vectorized Texas Hold'em environment",
    .tp_basicsize = sizeof(PokerBatchEnvObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = PokerBatchEnvObject_new,
    .tp_init = (initproc)PokerBatchEnvObject_init,
    .tp_dealloc = (destructor)PokerBatchEnvObject_dealloc,
    .tp_methods = PokerBatchEnvObject_methods,
    .tp_getset = PokerBatchEnvObject_getsetters,
};

// ============================================================================
// Module Definition
// ============================================================================

static PyObject* poker_c_get_obs_size(PyObject* self, PyObject* args) {
    return PyLong_FromLong(OBS_SIZE);
}

static PyObject* poker_c_get_num_actions(PyObject* self, PyObject* args) {
    return PyLong_FromLong(NUM_ACTIONS);
}

static PyMethodDef poker_c_methods[] = {
    {"get_obs_size", poker_c_get_obs_size, METH_NOARGS, "Get observation size"},
    {"get_num_actions", poker_c_get_num_actions, METH_NOARGS, "Get number of actions"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef poker_c_module = {
    PyModuleDef_HEAD_INIT,
    "poker_c",
    "High-performance Texas Hold'em Poker in C",
    -1,
    poker_c_methods
};

PyMODINIT_FUNC PyInit_poker_c(void) {
    import_array();

    PyObject* m = PyModule_Create(&poker_c_module);
    if (m == NULL) return NULL;

    if (PyType_Ready(&PokerBatchEnvType) < 0) return NULL;

    Py_INCREF(&PokerBatchEnvType);
    if (PyModule_AddObject(m, "PokerBatchEnv", (PyObject*)&PokerBatchEnvType) < 0) {
        Py_DECREF(&PokerBatchEnvType);
        Py_DECREF(m);
        return NULL;
    }

    PyModule_AddIntConstant(m, "OBS_SIZE", OBS_SIZE);
    PyModule_AddIntConstant(m, "NUM_ACTIONS", NUM_ACTIONS);

    return m;
}
