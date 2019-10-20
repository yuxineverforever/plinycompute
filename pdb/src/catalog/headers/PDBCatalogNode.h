#include <utility>

//
// Created by dimitrije on 9/4/18.
//

#ifndef PDB_PDBCATALOGNODE_H
#define PDB_PDBCATALOGNODE_H

#include <sqlite_orm.h>

namespace pdb {

/**
 * This is just a definition for the shared pointer on the type
 */
class PDBCatalogNode;
typedef std::shared_ptr<PDBCatalogNode> PDBCatalogNodePtr;

class PDBCatalogNode {

 public:

  /**
   * The default constructor needed by the orm
   */
  PDBCatalogNode() = default;

  /**
   * The initialize constructor
   * @param nodeID - the identifier of the node it should be created in the format of "ip:port"
   * @param address - the ip address of the node
   * @param port - the port of the node
   * @param nodeType - and the type of the node { worker, master } I guess
   */
  PDBCatalogNode(std::string nodeID, std::string address, int port, std::string nodeType, int32_t numCores, int64_t totalMemory, bool active)
      : nodeID(std::move(nodeID)), address(std::move(address)), port(port), nodeType(std::move(nodeType)), numCores(numCores), totalMemory(totalMemory), active(active) {}


  /**
   * The id of the node is a combination of the ip address and the port concatenated by a column
   */
  std::string nodeID;

  /**
   * The ip address of the node
   */
  std::string address;

  /**
   * The port of the node
   */
  int port;

  /**
   * The node type
   */
  std::string nodeType;

  /**
   * The number of cores on the node
   */
  int32_t numCores = -1;

  /**
   * The amount of memory on the node
   */
  int64_t totalMemory = -1;

  /**
   * True if the node is active false otherwise
   */
  bool active = false;

  /**
   * Return the schema of the database object
   * @return the schema
   */
  static auto getSchema() {

    // return the schema
    return sqlite_orm::make_table("nodes", sqlite_orm::make_column("nodeID", &PDBCatalogNode::nodeID),
                                           sqlite_orm::make_column("nodeAddress", &PDBCatalogNode::address),
                                           sqlite_orm::make_column("nodePort", &PDBCatalogNode::port),
                                           sqlite_orm::make_column("nodeType", &PDBCatalogNode::nodeType),
                                           sqlite_orm::make_column("nodeNumCores", &PDBCatalogNode::numCores),
                                           sqlite_orm::make_column("nodeTotalMemory", &PDBCatalogNode::totalMemory),
                                           sqlite_orm::make_column("nodeActive", &PDBCatalogNode::active),
                                           sqlite_orm::primary_key(&PDBCatalogNode::nodeID));
  }

};

}

#endif //PDB_PDBCATALOGNODE_H
