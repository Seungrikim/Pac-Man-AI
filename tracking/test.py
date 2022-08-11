            prevGhostPositions = list(oldParticle)
            for i in range(self.numGhosts):
                if (oldParticle, i) in cache:
                    newParticle[i] = cache[(oldParticle, i)].sample()
                else:
                    newPosDist = self.getPositionDistribution(gameState, prevGhostPositions, i, self.ghostAgents[i])
                    cache[(oldParticle, i)] = newPosDist
                    newParticle[i] = newPosDist.sample()



            for i , part in enumerate(newParticle):
                newParticle[i] = self.getPositionDistribution(gameState, oldParticle, i, self.ghostAgents[i]).sample()